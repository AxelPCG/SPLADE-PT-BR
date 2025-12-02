# SPLADE-PT-BR

<div align="center">

[![HuggingFace](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/AxelPCG/splade-pt-br)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**SPLADE sparse retrieval model trained for Brazilian Portuguese**

[Model Card](https://huggingface.co/AxelPCG/splade-pt-br) • [Usage Guide](docs/USAGE.md) • [Training](#-training) • [Results](#-model-details--results)

</div>

---

## 📌 Overview

SPLADE-PT-BR is a sparse neural retrieval model optimized for **Brazilian Portuguese** text search. Based on [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) and trained on Portuguese question-answering datasets, it produces interpretable sparse vectors perfect for RAG systems and semantic search.

### Why SPLADE-PT-BR?

- 🎯 **Native Portuguese**: Trained on BERTimbau with Portuguese-specific vocabulary
- ⚡ **Fast & Efficient**: ~99.5% sparse vectors enable inverted index search
- 🔍 **Semantic Expansion**: Automatically expands queries with related terms
- 🛠️ **Easy Integration**: Works with any vector database or custom retrieval systems
- 📊 **High Quality**: 150K training iterations, final loss: 0.000047

---

## 🚀 Quick Start

### Installation

```bash
# Install system dependencies
sudo apt-get update && sudo apt-get install -y python3.11-dev build-essential

# Install Python dependencies
uv sync
```

### Load Model

```python
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade

model = Splade.from_pretrained("AxelPCG/splade-pt-br")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
```

### Encode Text

```python
import torch

# Encode query
query = "Qual é a capital do Brasil?"
query_tokens = tokenizer(query, return_tensors="pt", max_length=256, truncation=True)

with torch.no_grad():
    query_vec = model(q_kwargs=query_tokens)["q_rep"].squeeze()

# Get sparse representation
indices = torch.nonzero(query_vec).squeeze().tolist()
values = query_vec[indices].tolist()

print(f"Sparsity: {len(indices)} / {query_vec.shape[0]} dimensions")
# Output: ~120 / 29794 dimensions (~99.6% sparse)
```

For complete examples including retrieval, see [USAGE.md](docs/USAGE.md).

---

## 📊 Model Details & Results

| Metric | Value |
|--------|-------|
| **Base Model** | BERTimbau (neuralmind/bert-base-portuguese-cased) |
| **Training Dataset** | mMARCO Portuguese (unicamp-dl/mmarco) |
| **Validation Dataset** | mRobust (unicamp-dl/mrobust) |
| **Iterations** | 150,000 |
| **Final Loss** | 0.000047 |
| **Vocabulary Size** | 29,794 |
| **Sparsity** | ~99.5% (100-150 active dims) |

### Evaluation Results

**Dataset**: mRobust (TREC Robust04 Portuguese)
- 528,032 documents
- 250 queries
- Evaluation date: 2025-12-02

#### SPLADE-PT-BR Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **MRR@10** | **0.453** | Mean Reciprocal Rank - First relevant doc at position ~2.2 |

#### Comparison: SPLADE-PT-BR vs SPLADE-EN

Performance on Portuguese dataset (mRobust - 528k docs, 250 queries):

| Model | Language | Base Model | MRR@10 | Performance |
|-------|----------|------------|--------|-------------|
| **SPLADE-PT-BR** | Portuguese | BERTimbau | **0.453** | **+18.3% better** |
| SPLADE-EN | English | BERT-EN | 0.383 | Baseline |

**Key Findings:**
- ✅ **SPLADE-PT-BR is 18.3% better** than SPLADE-EN on Portuguese queries
- ✅ Native Portuguese training (BERTimbau + mMARCO-PT) significantly improves retrieval quality
- ✅ MRR@10 of 0.453 means first relevant document appears at position ~2.2 on average

**Interpretation**: The Portuguese-adapted model demonstrates **clear superiority** over the English model for Portuguese IR tasks, validating the importance of language-specific training.

> 📊 For detailed evaluation metrics and comparison results, see [scripts/evaluation/README.md](scripts/evaluation/README.md) or `evaluation_results/comparison_report_*.json`

---

## 🔬 Training

<a name="training"></a>

### Configuration

```yaml
Base Model: neuralmind/bert-base-portuguese-cased
Training Data: mMARCO Portuguese (unicamp-dl/mmarco)
Validation Data: mRobust (unicamp-dl/mrobust)
Iterations: 150,000
Batch Size: 8 (effective: 32 with gradient accumulation)
Learning Rate: 2e-5
Regularization: FLOPS (λ_q=0.0003, λ_d=0.0001)
Mixed Precision: FP16
```

### Run Training

**Using Training Script (Recommended):**

```bash
# Full training pipeline
python scripts/training/train_splade_pt.py

# Skip completed steps
python scripts/training/train_splade_pt.py --skip-setup --skip-download
```

**Using Jupyter Notebook:**

The training notebook is available in `notebooks/SPLADE_v2_PTBR_treinamento.ipynb`.

**Manual Training:**

```bash
cd splade
SPLADE_CONFIG_NAME=config_splade_pt python3 -m splade.train_from_triplets_ids
```

### Important Notes

- The `splade/` directory is **not included** in this repository
- It is automatically cloned from `https://github.com/leobavila/splade.git` during training
- Necessary patches (AdamW, lazy loading, memory optimizations) are applied automatically
- This keeps the repository clean and ensures you get the latest SPLADE code with Portuguese-specific patches

---

## 📈 Evaluation

```bash
# Run evaluation
python scripts/evaluation/run_evaluation_comparator.py --pt-only
```

Results saved to `evaluation_results/` with detailed metrics and execution summary.

---

## 📁 Project Structure

```
SPLADE-PT-BR/
├── docs/USAGE.md                # Usage guide
├── notebooks/                   # Training notebook
├── scripts/
│   ├── training/train_splade_pt.py
│   └── evaluation/run_evaluation_comparator.py
├── evaluation_results/          # Evaluation metrics
└── splade/                      # Auto-cloned during training
```

---

## 🙏 Acknowledgments

- **SPLADE** by NAVER Labs ([naver/splade](https://github.com/naver/splade)) and [leobavila/splade](https://github.com/leobavila/splade) fork
- **BERTimbau** by Neuralmind
- **mMARCO & mRobust Portuguese** by UNICAMP-DL
- **Quati Dataset** research ([Bueno et al., 2024](https://arxiv.org/abs/2404.06976)) - Inspiration for native Portuguese IR

---

## 📚 Citation

```bibtex
@misc{splade-pt-br-2025,
  author = {Axel Chepanski},
  title = {SPLADE-PT-BR: Sparse Retrieval for Portuguese},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/AxelPCG/splade-pt-br}
}
```

---

## 📄 License

Apache 2.0 License

---

<div align="center">
  
**[View on Hugging Face](https://huggingface.co/AxelPCG/splade-pt-br)** • **[Usage Guide](docs/USAGE.md)**

</div>
