# 🛠️ Scripts

Utility and training scripts for SPLADE-PT-BR.

## Training

**`training/train_splade_pt.py`** - Complete training pipeline

```bash
python scripts/training/train_splade_pt.py [--skip-setup] [--skip-download] [--skip-qrel] [--skip-config]
```

## Utilities

**`utils/upload_to_hf.py`** - Upload model to HuggingFace
```bash
python scripts/utils/upload_to_hf.py
```

**`utils/compare_models.py`** - Compare with original SPLADE
```bash
python scripts/utils/compare_models.py
```

**`utils/visualize_results.py`** - Generate visualizations
```bash
python scripts/utils/visualize_results.py
```

## Setup

- `setup.sh` - Install system dependencies
- `setup_env.sh` - Configure environment variables
