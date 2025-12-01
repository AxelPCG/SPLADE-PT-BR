# 🛠️ Scripts Directory

This directory contains all utility and training scripts for the SPLADE-PT-BR project.

## 📂 Directory Structure

### `training/`
Training-related scripts:
- **`train_splade_pt.py`** - Main training script with modular workflow
  - Handles setup, data download, QREL conversion, config generation, and training
  - Supports CLI flags to skip specific steps
  - Usage: `python scripts/training/train_splade_pt.py [options]`

### `utils/`
Utility scripts for model management and analysis:
- **`upload_to_hf.py`** - Upload trained model to HuggingFace Hub
  - Prepares model files and uploads to HF repository
  - Usage: `python scripts/utils/upload_to_hf.py`

- **`compare_models.py`** - Compare SPLADE-PT-BR with original SPLADE
  - Generates comparison metrics and reports
  - Usage: `python scripts/utils/compare_models.py`

- **`visualize_results.py`** - Visualize training results
  - Creates graphs for loss curves, sparsity, and metrics
  - Usage: `python scripts/utils/visualize_results.py`

### Setup Scripts
Environment setup scripts:
- **`setup.sh`** - Install system dependencies (Python 3.11 headers, etc.)
- **`setup_env.sh`** - Configure environment variables and `.env` file

## 🚀 Quick Start

### Training
```bash
# Full training pipeline
python scripts/training/train_splade_pt.py

# Skip already completed steps
python scripts/training/train_splade_pt.py --skip-setup --skip-download
```

### Model Upload
```bash
# Upload to HuggingFace
python scripts/utils/upload_to_hf.py
```

### Analysis
```bash
# Compare with original SPLADE
python scripts/utils/compare_models.py

# Generate visualizations
python scripts/utils/visualize_results.py
```

## 📝 Notes

- All scripts should be run from the project root directory
- Scripts use `PROJECT_ROOT = Path(__file__).parent.parent` to locate project files
- Environment variables should be configured via `scripts/setup_env.sh` before running

