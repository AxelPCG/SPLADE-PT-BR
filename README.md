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

## 📚 Dataset Split Methodology

### SPLADE-PT-BR (This Model)

**Training Phase:**
- **Dataset**: [mMARCO Portuguese](https://huggingface.co/datasets/unicamp-dl/mmarco) (`unicamp-dl/mmarco`)
  - **Corpus**: `portuguese_collection.tsv` (~8.8M documents)
  - **Training Queries**: `portuguese_queries.train.tsv` (training queries)
  - **Triplets**: `triples.train.ids.small.tsv` (query-positive doc-negative doc triplets)
- **Base Model**: BERTimbau (`neuralmind/bert-base-portuguese-cased`)
- **Purpose**: Learn Portuguese-specific semantic expansion and term weighting

**Validation Phase (during training):**
- **Dataset**: [mRobust](https://huggingface.co/datasets/unicamp-dl/mrobust) (`unicamp-dl/mrobust`)
  - Used for validation checkpoints during training
  - Ensures the model generalizes to unseen Portuguese data

**Test/Evaluation Phase:**
- **Dataset**: mRobust (TREC Robust04 Portuguese translation)
  - **Documents**: 528,032 Portuguese documents
  - **Queries**: 250 test queries in Portuguese
  - **QRELs**: Relevance judgments (which docs are relevant for each query)
- **Purpose**: Final evaluation on completely unseen data

**✅ No Data Leakage**: Training (mMARCO) and testing (mRobust) are **completely different datasets** with different documents and queries, ensuring valid evaluation.

---

### SPLADE-EN (Original NAVER Model)

**Training Phase:**
- **Dataset**: [MS MARCO](https://microsoft.github.io/msmarco/) (English)
  - **Corpus**: ~8.8M English documents
  - **Training Queries**: ~500k English queries
  - **Triplets**: Query-positive-negative triplets in English
- **Base Model**: BERT-base-uncased (English vocabulary)
- **Model**: `naver/splade-cocondenser-ensembledistil`

**Original Test Phase:**
- **Datasets**: BEIR, MS MARCO Dev, TREC (all in English)
- Evaluated on standard English IR benchmarks

**Cross-Lingual Test (in this project):**
- **Dataset**: mRobust Portuguese (same as SPLADE-PT-BR test)
- **Purpose**: Compare English model performance on Portuguese data
- **Result**: MRR@10 = 0.383 (significantly lower than PT-BR's 0.453)

---

### Dataset Comparison Table

| Aspect | SPLADE-EN | SPLADE-PT-BR |
|--------|-----------|--------------|
| **Training Dataset** | MS MARCO (English) | mMARCO (Portuguese) |
| **Training Corpus Size** | ~8.8M docs (EN) | ~8.8M docs (PT) |
| **Validation Dataset** | MS MARCO Dev (EN) | mRobust (PT) |
| **Test Dataset** | BEIR/TREC (EN) | mRobust (PT) |
| **Base Model** | BERT-base-uncased | BERTimbau |
| **Vocabulary** | ~30k tokens (EN) | 29,794 tokens (PT) |
| **Cross-Lingual Test** | mRobust (PT) ⚠️ | mRobust (PT) ✅ |
| **MRR@10 on PT data** | 0.383 | **0.453** (+18.3%) |

**Key Insight**: The +18.3% performance improvement demonstrates that **native language training** is crucial for optimal retrieval quality. Cross-lingual models (EN→PT) significantly underperform compared to models trained directly on Portuguese data.

---

### Data Directory Structure

```
splade/data/pt/
├── triplets/              # Training Data (mMARCO)
│   ├── corpus.tsv         # 8.8M Portuguese documents
│   ├── queries_train.tsv  # Training queries
│   └── raw.tsv            # Query-doc-doc triplets
│
└── val_retrieval/         # Test Data (mRobust)
    ├── collection/
    │   └── raw.tsv        # 528k test documents
    ├── queries/
    │   └── raw.tsv        # 250 test queries
    └── qrel.json          # Relevance judgments
```

**Download & Setup**: All datasets are automatically downloaded by `scripts/training/train_splade_pt.py` from HuggingFace Hub.

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
