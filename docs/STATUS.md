# SPLADE-PT-BR - Project Status

## ✅ Trained Model

- **Checkpoint**: `splade/experiments/pt/checkpoint/model_ckpt/model_final_checkpoint.tar`
- **Size**: 1.2 GB
- **Parameters**: 131,866,052
- **Iterations**: 150,000
- **Final Loss**: 0.000000048 (excellent convergence)

## ✅ Configuration

- **Base Model**: neuralmind/bert-base-portuguese-cased (BERTimbau)
- **Batch Size**: 8 (effective: 32 with gradient accumulation)
- **Learning Rate**: 2e-5
- **FLOPS Regularization**: λ_q=0.0003, λ_d=0.0001
- **Max Length**: 256 tokens
- **FP16**: True

## ✅ Validation Data

- **Dataset**: mRobust (`unicamp-dl/mrobust`) - TREC Robust04 translated to Portuguese
- **Documents**: 528,032 in `splade/data/pt/val_retrieval/collection/`
- **Queries**: 250 in `splade/data/pt/val_retrieval/queries/`
- **QRELs**: Present in `qrel.json`

## ✅ Functional Scripts

1. **`scripts/utils/upload_to_hf.py`** - Upload to HuggingFace ✓
2. **`scripts/utils/compare_models.py`** - Comparison with original SPLADE ✓
3. **`scripts/utils/visualize_results.py`** - Graphs and visualizations ✓

## ✅ Generated Visualizations

- **training_loss.png** - Loss curve (150k iterations)
- **sparsity_info.png** - Sparsity information

## 📦 Hugging Face

- **Username**: AxelPCG
- **Model Name**: splade-pt-br
- **URL**: https://huggingface.co/AxelPCG/splade-pt-br
- **Status**: ✅ **UPLOAD COMPLETED** (1.31 GB transferred)

## 📝 Documentation

- **README.md** - Main project documentation
- **docs/USAGE.md** - Complete usage guide (generic for any vector database)
- **docs/MODEL_CARD.md** - HuggingFace model card

## 🎯 Model Usage

### Load from HuggingFace

```python
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade

# Load public model
model = Splade.from_pretrained("AxelPCG/splade-pt-br")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Use
query = "Qual a capital do Brasil?"
tokens = tokenizer(query, return_tensors="pt", max_length=256, truncation=True)
result = model(q_kwargs=tokens)
print(f"Shape: {result['q_rep'].shape}")  # (1, 29794)
```

### (Optional) Evaluate Performance

```bash
cd splade
source ../.venv/bin/activate

# Index documents
python -m splade.index config=config_splade_pt

# Perform retrieval
python -m splade.retrieve config=config_splade_pt

# Compare and visualize
cd ..
python scripts/utils/compare_models.py
python scripts/utils/visualize_results.py
```

## 📊 Useful Commands

```bash
# Check upload progress
tail -f upload.log

# Test model locally
cd splade
python -c "from splade.models.transformer_rep import Splade; model = Splade('neuralmind/bert-base-portuguese-cased'); print('OK')"

# After upload, load from HuggingFace
python -c "from splade.models.transformer_rep import Splade; model = Splade.from_pretrained('AxelPCG/splade-pt-br'); print('OK')"
```

---

**Status**: ✅ Model ready and public on HuggingFace Hub
**Last update**: 2025-12-01 12:35

