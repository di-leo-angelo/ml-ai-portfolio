# Auto-eval summary — Qwen3.8-27B + LoRA adapter (test set, n=230)

Method: greedy generation (max_new_tokens=512). Exact match = final-answer
equivalence for math subjects (fractions compared exactly; 3/36 == 1/12).
Judge = Qwen3-32B-AWQ rubric scores merged from judge_scores.jsonl (if present).

| subject | n | exact match | steps % | under ~350 words % | median words | median gen (s) |
| --- | --- | --- | --- | --- | --- | --- |
| algebra | 23 | 61% (14/23) | 4% | 100% | 202 | 39 |
| biology | 23 | - | 52% | 91% | 288 | 42 |
| calculus | 23 | 52% (12/23) | 0% | 100% | 122 | 41 |
| chemistry | 23 | - | 70% | 100% | 256 | 42 |
| computer_science | 23 | 13% (3/23) | 26% | 91% | 291 | 35 |
| english_literature | 23 | - | 9% | 65% | 339 | 42 |
| history | 23 | - | 74% | 78% | 327 | 42 |
| physics | 23 | 39% (9/23) | 0% | 100% | 137 | 26 |
| probability | 23 | 61% (14/23) | 4% | 100% | 132 | 23 |
| world_geography | 23 | - | 65% | 96% | 322 | 42 |
| **overall** | **230** | **45% (52/115)** | 30% | 92% | | |

## LLM-judge scores (1-5)

| subject | n | correctness | format | completeness |
| --- | --- | --- | --- | --- |
| algebra | 23 | 5.00 | 4.61 | 4.83 |
| biology | 23 | 4.91 | 4.30 | 4.70 |
| calculus | 23 | 5.00 | 4.65 | 4.87 |
| chemistry | 23 | 5.00 | 4.87 | 5.00 |
| computer_science | 21 | 5.00 | 4.43 | 4.90 |
| english_literature | 19 | 4.53 | 3.79 | 4.16 |
| history | 20 | 4.75 | 3.75 | 4.20 |
| physics | 23 | 5.00 | 4.87 | 5.00 |
| probability | 22 | 5.00 | 4.82 | 4.91 |
| world_geography | 23 | 4.74 | 3.91 | 4.04 |
