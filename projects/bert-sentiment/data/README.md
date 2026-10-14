# Dataset

This project uses **SST-2** (Stanford Sentiment Treebank, binary): 67,349
movie-review sentences labeled positive or negative for sentiment
classification.

- Source: Socher et al., *Recursive Deep Models for Semantic Compositionality
  Over a Sentiment Bank* (ACL 2013), via the GLUE benchmark
- Hugging Face: <https://huggingface.co/datasets/glue> (`glue/sst2`)
- License: the SST data is released for non-commercial research and
  educational use; see the dataset card for the full terms

Download the data reproducibly from the repository root:

```powershell
python projects/bert-sentiment/scripts/download_data.py
```

This creates `data/sst2/` (on-disk Arrow cache, not committed) and validates
the expected row counts (67,349 train / 872 validation / 1,821 test) and
label schema.

## Important data notes

- Labels: `negative` (0) / `positive` (1); the train split is imbalanced:
  29,780 negative / 37,569 positive (43.9% / 56.1%), so a trivial
  always-positive baseline scores 56.1%.
- Text lives in the `sentence` column (GLUE convention).
- The 1,821-row **test split has no public labels** (the labels in the Hub
  mirror are the sentinel value `-1`; the GLUE evaluation server holds the
  real ones), so local evaluation is reported on the 872-row validation
  (dev) split — the same convention Hugging Face uses when reporting GLUE
  benchmark numbers.
- Sentences are short (movie review fragments): the longest training
  sentence is 52 words, and 99.96% fit in 64 WordPiece tokens, so the
  project default is `max_length=64` (negligible truncation, ~2x CPU
  throughput vs. the usual 128).
