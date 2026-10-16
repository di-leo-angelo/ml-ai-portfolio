# BERT Sentiment Classification (SST-2)

## 1. Problem and motivation

Sentiment classification on **SST-2** (Stanford Sentiment Treebank, binary),
the standard text-classification benchmark in the GLUE suite. The goal here
is not the best published number but a complete, modern fine-tuning protocol
end-to-end on CPU-only hardware:

- fine-tuning a pre-trained Transformer with the Hugging Face `Trainer`,
- best-checkpoint selection on the dev split (accuracy), with early stopping,
- honest evaluation (the GLUE test split has no public labels, so the
  reported metric is the official dev split — the same convention Hugging
  Face uses for GLUE reporting),
- a documented ablation between BERT-base and DistilBERT,
- publishing both models to the Hugging Face Hub.

Two base models are supported via `--model`:

| Name | Hub ID | Params | Role |
|---|---|---|---|
| `bert-base` (default) | `bert-base-uncased` | 110M | Headline model, full run |
| `distilbert` | `distilbert-base-uncased` | 66M | Fast path (`--quick`) and ablation |

On this hardware (no GPU), DistilBERT is the pragmatic choice for iteration;
the full BERT-base run documents the quality/compute tradeoff.

## 2. Dataset

**SST-2**: 67,349 movie-review sentences labeled `negative`/`positive`,
plus an 872-row validation (dev) split. See [data/README.md](data/README.md)
for source, license, and download instructions.

- Source: Socher et al. (ACL 2013), via the GLUE benchmark
  (`glue/sst2` on the Hugging Face Hub)
- Terms: non-commercial research and educational use
- Local layout: `data/sst2/` (not committed; re-download with
  `scripts/download_data.py`)

The 1,821-row GLUE test split has no public labels (evaluation server), so
it is downloaded and validated but never used locally.

## 3. Methodology

| Component | Choice |
|---|---|
| Base model | `bert-base-uncased` (default) or `distilbert-base-uncased` |
| Preprocessing | WordPiece tokenizer, `max_length=64`, truncation (99.96% of sentences fit; longest is 52 words) |
| Fine-tuning | full-parameter, Hugging Face `Trainer` |
| Optimizer | AdamW (Trainer defaults), lr 2e-5, weight decay 0.01 |
| Schedule | linear warmup for 10% of steps, then linear decay |
| Batch size | 16 train / 32 eval |
| Early stopping | patience 3 epochs on dev accuracy; best-dev model restored |
| Evaluation | dev accuracy + binary F1, confusion matrix, error examples |
| Seed | 42 |

## 4. Main results

Full runs on CPU (Intel i5-8250U, no GPU), seed 42. Both runs use the full
protocol (3 epochs, best-dev checkpoint, early stopping patience 3); both
models peak at epoch 1 (BERT-base holds 0.9243 at epochs 1 and 3; DistilBERT
declines 0.9014 → 0.8979 → 0.8968), so the saved checkpoint is the epoch-1
model in both cases.

| Model | Params | Train samples | Epochs | Dev accuracy | Dev F1 | Training time |
|---|---|---|---|---|---|---|
| BERT-base (headline) | 110M | 67,349 | 3 | 0.9243 | 0.9263 | 11 h 27 m |
| DistilBERT (ablation) | 66M | 67,349 | 3 | 0.9014 | 0.9027 | 5 h 55 m |

Ablation read: DistilBERT gives up 2.3 dev-accuracy points (2.4 F1 points)
for 40% fewer parameters, ~2× faster training, and ~2× faster inference.
BERT-base dev errors: 66/872 (37 negatives and 29 positives misclassified),
dominated by irony/ambiguity fragments.

`--quick` (pipeline smoke test, 4,096 samples, 1 epoch, DistilBERT): dev
accuracy **0.8544**, dev F1 **0.8612**, 11.8 min wall time.

## 5. Limitations and possible improvements

- CPU-only training: a GPU would cut the full run from hours to minutes.
- No hyperparameter search beyond chosen defaults (LR range, sequence
  length, batch size).
- Single fixed dev split: the GLUE test labels are not public, and a
  local K-fold split of the train data would give a more robust error bar.
- Possible extensions: multistage SST-5 classification, LoRA/QLoRA to
  reduce memory and training time, or an ensemble of BERT + DistilBERT.

## 6. Reproduction

From the repository root (Windows PowerShell):

```powershell
. .venv\Scripts\Activate.ps1
cd projects\bert-sentiment\src

python -m sentiment.train --quick    # smoke test: DistilBERT, 4k subset, 1 epoch (~15 min)
python -m sentiment.train            # headline run: BERT-base, full data, 3 epochs (~12-15 h on this CPU)
python -m sentiment.train --model distilbert   # ablation run: DistilBERT, full data, 3 epochs (~8-9 h)

python -m sentiment.evaluate         # dev report for models/bert-base (confusion matrix, errors)
python -m sentiment.evaluate --model distilbert
```

Unit tests (from the repository root):

```powershell
python -m pytest projects/bert-sentiment/tests -v
```

Outputs: `models/<model>/` (weights + tokenizer + `training_state.json` +
`training_history.json`), `runs/<model>/` (Trainer checkpoints),
`reports/figures/learning_curve_<model>.png`.

## 7. Publishing to the Hugging Face Hub

Both models are published under `di-leo-angelo` on the Hub
(`sst2-sentiment-bert-base`, `sst2-sentiment-distilbert`). The token lives in
the git-ignored `.env.local` at the repository root (`HF_TOKEN=...`).

```powershell
# Dry run (default): verifies the token and shows what would be uploaded
python projects/bert-sentiment/scripts/publish.py --model bert-base

# Actual upload (creates or updates the Hub repository)
python projects/bert-sentiment/scripts/publish.py --model bert-base --push
```

The publisher writes a model card (metrics, training details, usage snippet)
into the checkpoint directory and uploads it together with the weights, so
the Hub repository renders a full README and an Inference API widget:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification", model="di-leo-angelo/sst2-sentiment-bert-base"
)
print(classifier("The film was a fantastic, hilarious ride."))
```
