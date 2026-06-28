# concept_space

This repository hosts scripts for the paper [A framework for analyzing concept representations in neural models](https://arxiv.org/abs/2605.01381)

The code estimates and evaluates concept spaces for the frame-level phone and speaker concepts, using HuBERT representations on Librispeech dev and test clean sets.

Supported concept estimators: LEACE, COV, MLR, CPCA, LDA, Random, and Identity (ambient space).

## Installation

```bash
pip install -e .
```

## Setup

(Optional) Update .env
```dotenv
# Directory containing forced-alignment .ali files
# Expected files: dev-clean.ali, test-clean.ali
ALIGNMENT_DIR=/path/to/forced_alignment

# Directory for model inference cache (default: cache)
CACHE_DIR=cache

# Directory for cached speaker-overlap train/test splits
SPK_TEST_SPLIT_DIR=save_spk_test_split

# Evaluation hyperparameters
EVAL_N_JOBS=8
EVAL_MAX_ITER=200
DATA_N_JOBS=4
```

The scripts load `.env` automatically at startup via `python-dotenv`.

## Running

### Per-layer evaluation (Section 5.1)

Trains all estimators on `dev-clean` features and evaluates phone/speaker probes across layers on `test-clean`:

```bash
python run_acl_test_suite.py <log_dir> [--layers 0 1 ... 12]
```

Outputs one directory per layer: `<log_dir>/layer_{N}/`.
This per-layer evaluation reads and writes feature caches under `CACHE_DIR` by default.

### Train-clean-460 LEACE evaluation (Section 5.2)

Train standard LEACE spaces on `train-clean-100` plus `train-clean-360`, then evaluate on `test-clean`:

```bash
python run_leace_train_clean460.py <score_dir> [--layer 11] [--cache-train]
```

By default, train-clean-100/360 features are computed on-the-fly without using caches. Use `--cache-train` to store cache.

This saves LEACE spaces under `save_space/<model_name>/<layer>/` and writes score CSVs under `<score_dir>/`. The script will automatically load LEACE space weights if found.

### Summarizing results

Compute the retention, purity, leakage, and interference summary scores:

```bash
python gather_results.py score <log_dir> [--layers 11]  [--save-dir <score_dir>]
```

This writes `raw_score-v2_{mode}.csv` files containing paper-style scores to `<score_dir>`.


Use `gather` to get raw accuracy scores for each classifiers and estimators in one table:

```bash
python gather_results.py gather <log_dir> [--layers 11] [--save-dir <score_dir>]
```

For both commands, `<log_dir>` should match the argument passed to `run_acl_test_suite.py`.

### Plotting figures

Scripts for plotting figures can be foud in `plot_scripts/`.

## Output structure

Per-layer evaluation:
```
<log_dir>/
  layer_{N}/
    {proj_name}/
      rejection/
        dev-clean-phone.csv
        dev-clean-spk.csv
      projection/
        dev-clean-phone.csv
        dev-clean-spk.csv
```
Each CSV has columns: `proj name`, `proj_train_data`, `clf_train_data`, `clf_train_portion`, `dev-clean`, `test-clean`, `dev-clean-loss`, `test-clean-loss`.

## Data

LibriSpeech is downloaded automatically from the HuggingFace Hub (`openslr/librispeech_asr`). Extracted features are cached under `CACHE_DIR` (default: `cache/`), configured via `.env`. Plan for roughly the following cache sizes:

| Split | Expected cache size |
| --- | ---: |
| `dev-clean` | 40 GB |
| `test-clean` | 40 GB |
| `train-clean-100` | 700 GB (Optional)|
| `train-clean-360` | 2.5 TB (Optional, Expected) |

Forced-alignment files (`.ali`) must be provided separately and pointed to via `ALIGNMENT_DIR`. The pre-computed speaker test splits in `save_spk_test_split/` are included in the repo.
