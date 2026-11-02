"""Local HF transformers backend (the only backend that needs a GPU pod).

Mirrors P1's eval_run.py generation recipe so the dogfood numbers stay
comparable: 4-bit NF4 base (no double quantization, exactly as P1's eval),
``AutoModelForCausalLM`` + ``trust_remote_code`` (the path P1's eval used),
greedy decoding, the same chat-template call, and per-generation peak-VRAM
tracking (P1 convention: reset_peak_memory_stats + running max).

torch/transformers/peft are imported lazily so the CPU-only paths (tests,
API backends) never need the GPU stack.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

from .base import GenerationError, GenerationJob, GenerationRecord


class LocalHFBackend:
    """Generation backend that runs the model in-process (Hugging Face)."""

    kind = "local"

    def __init__(
        self,
        name: str,
        model_id: str,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        self.name = name
        self.model = model_id
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.peak_vram_gb: float | None = None
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        """Lazily load the tokenizer + model (optionally LoRA-wrapped) once."""
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if self.load_in_4bit
            else None
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=dtype_map[self.dtype],
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            print(f"[{self.name}] adapter active: {self.adapter_path}", flush=True)
        self._model.eval()
        print(f"[{self.name}] model loaded: {self.model_id}", flush=True)

    def _generate_one(self, job: GenerationJob) -> GenerationRecord:
        self._load()
        import torch

        prompt = self._tokenizer.apply_chat_template(
            job.messages(), tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        started = time.monotonic()
        with torch.inference_mode():
            generate_kwargs: dict = {
                "max_new_tokens": job.max_tokens,
                "do_sample": False,
            }
            if job.temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = job.temperature
                if job.seed:
                    generate_kwargs["seed"] = job.seed
            output_ids = self._model.generate(**inputs, **generate_kwargs)
        wall_s = time.monotonic() - started
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            peak = torch.cuda.max_memory_allocated() / 1e9
            self.peak_vram_gb = max(self.peak_vram_gb or 0.0, peak)
            vram_note = f"peak {self.peak_vram_gb:.1f}GB"
        else:
            vram_note = "cpu"
        answer = self._tokenizer.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()
        print(
            f"[{self.name}] {job.task_id} {wall_s:.1f}s {vram_note} :: {answer[:70]}",
            flush=True,
        )
        return GenerationRecord(
            task_id=job.task_id,
            condition=self.name,
            seed=job.seed,
            output=answer,
            backend="local",
            model=self.model_id,
            tokens_in=prompt_len,
            tokens_out=int(output_ids[0].shape[0]) - prompt_len,
            wall_s=round(wall_s, 3),
        )

    def generate(self, jobs: Sequence[GenerationJob]) -> Iterator[GenerationRecord]:
        for job in jobs:
            try:
                yield self._generate_one(job)
            except Exception as exc:
                raise GenerationError(
                    f"local generation failed for {job.task_id}: {exc}"
                ) from exc
