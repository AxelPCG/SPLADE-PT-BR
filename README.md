# SPLADE-PT-BR

<div align="center">

[![HuggingFace](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/AxelPCG/splade-pt-br)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**SPLADE sparse retrieval model trained for Brazilian Portuguese**

[Model Card](https://huggingface.co/AxelPCG/splade-pt-br) • [Usage Guide](docs/USAGE.md) • [Training Details](#training) • [Status](docs/STATUS.md)

</div>

---

## 📌 Overview

SPLADE-PT-BR is a sparse neural retrieval model optimized for **Brazilian Portuguese** text search. Based on [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) and trained on Portuguese question-answering datasets, it produces interpretable sparse vectors perfect for RAG systems and semantic search.

### Why SPLADE-PT-BR?

- 🎯 **Native Portuguese**: Trained on BERTimbau with Portuguese-specific vocabulary
- ⚡ **Fast & Efficient**: ~99% sparse vectors enable inverted index search
- 🔍 **Semantic Expansion**: Automatically expands queries with related terms
- 🛠️ **Easy Integration**: Works with any vector database or custom retrieval systems
- 📊 **High Quality**: 150K training iterations, final loss: 0.000047

### Quick Start

```bash
# Load from Hugging Face
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade

model = Splade.from_pretrained("AxelPCG/splade-pt-br")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
```

For detailed usage examples, see [USAGE.md](docs/USAGE.md).

---

## 📦 Installation

### Prerequisites

This project requires Python 3.11+ development headers to compile `pytrec-eval`:

```bash
sudo apt-get update
sudo apt-get install -y python3.11-dev build-essential
```

### Setup

#### Option 1: Automatic (Recommended)

   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

#### Option 2: Manual

   ```bash
   uv sync
   ```

### Verification

```bash
# Check pytrec-eval
python3.11 -c "import pytrec_eval; print('✅ pytrec-eval OK')"

# Check main dependencies
python3.11 -c "import torch; import transformers; print('✅ All dependencies OK')"
```

---

## 🚀 Using the Trained Model

### Download from Hugging Face

The trained model is available at: [`AxelPCG/splade-pt-br`](https://huggingface.co/AxelPCG/splade-pt-br)

```python
from splade.models.transformer_rep import Splade

model = Splade.from_pretrained("AxelPCG/splade-pt-br")
```

### Encode Text

```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Encode query
query = "Qual é a capital do Brasil?"
query_tokens = tokenizer(query, return_tensors="pt", max_length=256, truncation=True)

with torch.no_grad():
    query_vec = model(q_kwargs=query_tokens)["q_rep"].squeeze()

# Get sparse representation
indices = torch.nonzero(query_vec).squeeze().tolist()
values = query_vec[indices].tolist()

print(f"Sparsity: {len(indices)} / {query_vec.shape[0]} dimensions")
# Output: Sparsity: ~120 / 29794 dimensions (~99.6% sparse)
```

### Simple Retrieval Example

```python
# Build a simple inverted index
def create_inverted_index(documents):
    index = {}
    for doc_id, text in documents.items():
        tokens = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
        with torch.no_grad():
            vec = model(d_kwargs=tokens)["d_rep"].squeeze()
        
        indices = torch.nonzero(vec).squeeze().tolist()
        values = vec[indices].tolist()
        
        for idx, val in zip(indices, values):
            if idx not in index:
                index[idx] = []
            index[idx].append((doc_id, val))
    
    return index

# Search
def search(query, index, documents, top_k=5):
    tokens = tokenizer(query, return_tensors="pt", max_length=256, truncation=True)
    with torch.no_grad():
        query_vec = model(q_kwargs=tokens)["q_rep"].squeeze()
    
    q_indices = torch.nonzero(query_vec).squeeze().tolist()
    q_values = query_vec[q_indices].tolist()
    
    scores = {}
    for idx, q_val in zip(q_indices, q_values):
        if idx in index:
            for doc_id, d_val in index[idx]:
                scores[doc_id] = scores.get(doc_id, 0) + (q_val * d_val)
    
    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_id, documents[doc_id], score) for doc_id, score in results]

# Example
docs = {1: "Brasília é a capital do Brasil", 2: "Python é uma linguagem"}
index = create_inverted_index(docs)
results = search("capital brasileira", index, docs)
```

See [USAGE.md](docs/USAGE.md) for complete examples.

---

## 📊 Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Training Iterations** | 150,000 | Full convergence |
| **Final Loss** | 0.000047 | Excellent convergence |
| **Vocabulary Size** | 29,794 | Portuguese-optimized |
| **Sparsity** | ~99.5% | 100-150 active dims |
| **Base Model** | BERTimbau | neuralmind/bert-base-portuguese-cased |

*Evaluation metrics available after running evaluation pipeline (see below)*

---

## 🔬 Training

<a name="training"></a>

### Training Configuration

```yaml
Base Model: neuralmind/bert-base-portuguese-cased
Training Dataset: mMARCO Portuguese (unicamp-dl/mmarco)
Validation Dataset: mRobust (unicamp-dl/mrobust)
Iterations: 150,000
Batch Size: 8 (effective: 32 with gradient accumulation)
Learning Rate: 2e-5
Regularization: FLOPS (λ_q=0.0003, λ_d=0.0001)
Mixed Precision: FP16
```

### Run Training

#### Option 1: Using the Training Script (Recommended)

The modularized training script (`scripts/training/train_splade_pt.py`) provides a complete workflow:

```bash
# Full training workflow
python scripts/training/train_splade_pt.py
```

The script will:
1. Clone and patch the SPLADE repository
2. Download training and validation datasets (mMARCO and mRobust)
3. Convert QRELS to JSON format
4. Generate Hydra configuration files
5. Execute model training

**Skip completed steps:**
```bash
# Skip repository setup (already cloned and patched)
python scripts/training/train_splade_pt.py --skip-setup

# Skip dataset download (datasets already downloaded)
python scripts/training/train_splade_pt.py --skip-download

# Skip QREL conversion (already converted)
python scripts/training/train_splade_pt.py --skip-qrel

# Skip configuration generation (configs already exist)
python scripts/training/train_splade_pt.py --skip-config

# Only run training (everything else already done)
python scripts/training/train_splade_pt.py --skip-setup --skip-download --skip-qrel --skip-config
```

**Command line options:**
- `--skip-setup`: Skip repository cloning and patching
- `--skip-download`: Skip dataset download
- `--skip-qrel`: Skip QREL to JSON conversion
- `--skip-config`: Skip configuration file generation
- `--skip-training`: Skip training execution (only prepare environment)

#### Option 2: Manual Training

```bash
cd splade
python3 -m splade.train_from_triplets_ids +config=config_splade_pt
```

#### Option 3: Jupyter Notebook

The original training notebook is available in `notebooks/SPLADE_v2_PTBR_treinamento.ipynb` for interactive use.

**Note:** The notebook automatically detects the project root and applies necessary patches.

### Important Notes

- The `splade/` directory is **not included** in this repository
- It is automatically cloned from `https://github.com/leobavila/splade.git` when you run the training script or notebook
- Necessary compatibility patches (AdamW, lazy loading, memory optimizations) are applied automatically
- This approach keeps the repository clean and ensures you always get the latest SPLADE code with our Portuguese-specific patches

---

## 📈 Evaluation

### Run Complete Evaluation

```bash
# 1. Index documents
cd splade
python3 -m splade.index +config=config_splade_pt

# 2. Retrieve and calculate metrics
python3 -m splade.retrieve +config=config_splade_pt

# 3. Compare with original SPLADE
cd ..
python3 scripts/utils/compare_models.py

# 4. Generate visualizations
python3 scripts/utils/visualize_results.py
```

Results will be saved to `splade/experiments/pt/out/`.

---

## 📁 Project Structure

```
SPLADE-PT-BR/
├── splade/                       # Main SPLADE package
│   ├── conf/                     # Hydra configurations
│   │   ├── config_splade_pt.yaml
│   │   ├── train/config/splade_pt.yaml
│   │   ├── index/pt.yaml
│   │   └── retrieve_evaluate/pt.yaml
│   ├── splade/                   # Source code
│   │   ├── models/               # Model implementations
│   │   ├── tasks/                # Training/evaluation tasks
│   │   └── losses/               # Loss functions & regularization
│   ├── data/pt/                  # Portuguese datasets
│   │   ├── triplets/             # Training triplets
│   │   └── val_retrieval/        # Validation data
│   └── experiments/pt/           # Training outputs
│       ├── checkpoint/           # Model checkpoints
│       ├── index/                # Sparse indexes
│       └── out/                  # Evaluation results
├── notebooks/                    # Jupyter notebooks
│   └── SPLADE_v2_PTBR_treinamento.ipynb  # Training notebook
├── scripts/                      # All utility scripts
│   ├── training/                # Training scripts
│   │   └── train_splade_pt.py   # Modularized training script
│   ├── utils/                   # Utility scripts
│   │   ├── upload_to_hf.py      # Upload to HuggingFace
│   │   ├── compare_models.py    # Model comparison
│   │   └── visualize_results.py # Results visualization
│   ├── setup.sh                 # System dependencies setup
│   └── setup_env.sh             # Environment configuration
├── docs/                         # Documentation
│   ├── MODEL_CARD.md            # Hugging Face model card
│   ├── USAGE.md                 # Detailed usage guide
│   └── STATUS.md                # Training status and metrics
├── model_metadata.json           # Training metadata
├── main.py                      # Main entry point
└── README.md                    # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **SPLADE** original paper and implementation by Formal et al.
- **BERTimbau** by Neuralmind team
- **mMARCO Portuguese** and **mRobust Portuguese** datasets by UNICAMP-DL
- **Quati Dataset** - Inspiration from the work on native Portuguese IR datasets
- Hugging Face for model hosting

### Inspiration

This project was inspired by research on native Portuguese information retrieval datasets:
- **Quati Dataset**: Bueno et al. (2024) demonstrated the importance of native Portuguese IR datasets over translated ones for better capturing socio-cultural aspects of Brazilian Portuguese
- **UNICAMP-DL Research**: Their pioneering work on Portuguese NLP and information retrieval systems

---

## 📚 Citation

If you use this model, please cite:

```bibtex
@misc{splade-pt-br-2025,
  author = {Axel Chepanski},
  title = {SPLADE-PT-BR: Sparse Retrieval for Portuguese},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/AxelPCG/splade-pt-br}
}
```

### Related Work

This project builds upon research in Portuguese information retrieval:

```bibtex
@article{bueno2024quati,
  title={Quati: A Brazilian Portuguese Information Retrieval Dataset from Native Speakers},
  author={Bueno, Mirelle and de Oliveira, E. Seiti and Nogueira, Rodrigo and Lotufo, Roberto and Pereira, Jayr},
  journal={arXiv preprint arXiv:2404.06976},
  year={2024},
  url={https://arxiv.org/abs/2404.06976}
}

@inproceedings{bonifacio2021mmarco,
  title={mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset},
  author={Bonifacio, Luiz and Campiotti, Israel and Lotufo, Roberto and Nogueira, Rodrigo},
  booktitle={Proceedings of STIL 2021},
  year={2021},
  organization={SBC},
  url={https://sol.sbc.org.br/index.php/stil/article/view/31136}
}
```

Original SPLADE paper:
```bibtex
@inproceedings{formal2021splade,
  title={SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking},
  author={Formal, Thibault and Piwowarski, Benjamin and Clinchant, St{\'e}phane},
  booktitle={SIGIR 2021},
  year={2021}
}
```

---

<div align="center">
  
**[View on Hugging Face](https://huggingface.co/AxelPCG/splade-pt-br)** • **[Usage Guide](docs/USAGE.md)** • **[Status](docs/STATUS.md)**

</div>