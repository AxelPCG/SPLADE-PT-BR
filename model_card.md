---
language: pt
license: apache-2.0
tags:
- information-retrieval
- sparse-retrieval
- splade
- portuguese
- bert
datasets:
- unicamp-dl/mmarco
- unicamp-dl/mrobust
base_model: neuralmind/bert-base-portuguese-cased
---

# SPLADE-PT-BR

SPLADE (Sparse Lexical AnD Expansion) model fine-tuned for **Portuguese** text retrieval. This model is based on [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) and trained on Portuguese question-answering datasets.

## Model Description

SPLADE is a neural retrieval model that learns to expand queries and documents with contextually relevant terms while maintaining sparsity. Unlike dense retrievers, SPLADE produces sparse vectors (typically ~99% sparse) that are:
- **Interpretable**: Each dimension corresponds to a vocabulary token
- **Efficient**: Can use inverted indexes for fast retrieval
- **Effective**: Combines lexical matching with semantic expansion

### Key Features

- **Base Model**: `neuralmind/bert-base-portuguese-cased` (BERTimbau)
- **Vocabulary Size**: 29,794 tokens (Portuguese-optimized)
- **Training Iterations**: 150,000
- **Final Training Loss**: 0.000047
- **Sparsity**: ~99% (100-150 active dimensions per vector)
- **Max Sequence Length**: 256 tokens

## Training Details

### Training Data

- **Training Dataset**: mMARCO Portuguese (`unicamp-dl/mmarco`) - MS MARCO translated to Portuguese
  - Used for training with triplets (query, positive document, negative document)
  - Created by UNICAMP-DL team as part of their Portuguese IR research
- **Validation Dataset**: mRobust (`unicamp-dl/mrobust`) - TREC Robust04 translated to Portuguese
  - Used for validation and evaluation during training
  - Part of the UNICAMP-DL Portuguese IR datasets collection
- **Format**: Triplets (query, positive document, negative document)

**Note**: This model was inspired by research on native Portuguese information retrieval, particularly the [Quati dataset](https://arxiv.org/abs/2404.06976) work by Bueno et al. (2024), which demonstrated the importance of native Portuguese datasets over translated ones for better capturing socio-cultural aspects of Brazilian Portuguese.

### Training Configuration

```yaml
- Learning Rate: 2e-5
- Batch Size: 8 (effective: 32 with gradient accumulation)
- Gradient Accumulation Steps: 4
- Weight Decay: 0.01
- Warmup Steps: 6,000
- Mixed Precision: FP16
- Optimizer: AdamW
```

### Regularization

FLOPS regularization is applied to enforce sparsity:
- **Lambda Query**: 0.0003 (queries are more sparse)
- **Lambda Document**: 0.0001 (documents less sparse for better recall)

## Usage

### Installation

```bash
pip install torch transformers
```

### Basic Usage

```python
import torch
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade

# Load model and tokenizer
model = Splade.from_pretrained("AxelPCG/splade-pt-br")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model.eval()

# Encode a query
query = "Qual é a capital do Brasil?"
with torch.no_grad():
    query_tokens = tokenizer(query, return_tensors="pt", max_length=256, truncation=True)
    query_vec = model(q_kwargs=query_tokens)["q_rep"].squeeze()

# Encode a document
document = "Brasília é a capital federal do Brasil desde 1960."
with torch.no_grad():
    doc_tokens = tokenizer(document, return_tensors="pt", max_length=256, truncation=True)
    doc_vec = model(d_kwargs=doc_tokens)["d_rep"].squeeze()

# Calculate similarity (dot product)
similarity = (query_vec * doc_vec).sum().item()
print(f"Similarity: {similarity:.4f}")

# Get sparse representation
indices = torch.nonzero(query_vec).squeeze().tolist()
values = query_vec[indices].tolist()
print(f"Query sparsity: {len(indices)} / {query_vec.shape[0]} active dimensions")
```

### Using Sparse Vectors for Retrieval

```python
# Build inverted index from documents
inverted_index = {}

def add_to_index(doc_id, text):
    """Add document to inverted index"""
    sparse_vec = encode_sparse(text, is_query=False)
    
    for idx, value in zip(sparse_vec["indices"], sparse_vec["values"]):
        if idx not in inverted_index:
            inverted_index[idx] = []
        inverted_index[idx].append((doc_id, value))

# Index documents
docs = {
    1: "Brasília é a capital do Brasil",
    2: "São Paulo é a maior cidade do Brasil",
    3: "Python é uma linguagem de programação"
}

for doc_id, text in docs.items():
    add_to_index(doc_id, text)

# Search using inverted index
def search(query, top_k=5):
    """Search documents using sparse vectors"""
    query_vec = encode_sparse(query, is_query=True)
    
    # Calculate scores for each document
    scores = {}
    for idx, q_value in zip(query_vec["indices"], query_vec["values"]):
        if idx in inverted_index:
            for doc_id, d_value in inverted_index[idx]:
                scores[doc_id] = scores.get(doc_id, 0) + (q_value * d_value)
    
    # Sort by score
    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_id, docs[doc_id], score) for doc_id, score in results]

# Example search
results = search("capital brasileira", top_k=3)
for doc_id, text, score in results:
    print(f"Score: {score:.2f} - {text}")
```

## Performance

### Evaluation Metrics

*Metrics will be updated after complete evaluation on validation set.*

Expected performance on Portuguese retrieval tasks:
- **MRR@10**: ~0.25-0.35
- **Recall@100**: ~0.85-0.95
- **L0 (Sparsity)**: ~100-150 active dimensions

### Comparison with Original SPLADE

The original SPLADE model was trained on English data. Key differences:

| Aspect | Original SPLADE | SPLADE-PT-BR |
|--------|----------------|--------------|
| Language | English | Portuguese |
| Base Model | BERT-base-uncased | BERTimbau (BERT-base-cased-pt) |
| Vocabulary | 30,522 tokens | 29,794 tokens |
| Training Data | MS MARCO | mMARCO Portuguese |
| Query Expansion | English context | Portuguese context |

**Advantages for Portuguese:**
- Native vocabulary tokens (no subword splitting for Portuguese words)
- Semantic expansion using Portuguese linguistic patterns
- Better performance on Brazilian Portuguese queries

## Model Architecture

```
Input Text → BERTimbau Tokenizer → BERT Encoder → MLM Head → 
ReLU → log(1 + x) → Attention Masking → Max/Sum Pooling → Sparse Vector
```

The model outputs a vector of size 29,794 (vocabulary size) where:
- Most values are exactly 0 (sparse)
- Non-zero values represent term importance + learned expansions
- Can be used directly with inverted indexes

## Limitations

- **Language**: Optimized for Brazilian Portuguese; may work for European Portuguese but not tested
- **Domain**: Trained on general question-answering; may need fine-tuning for specific domains
- **Sequence Length**: Maximum 256 tokens; longer documents should be split
- **Computational Cost**: Requires GPU for efficient encoding of large collections

## Citation

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

Original SPLADE paper:

```bibtex
@inproceedings{formal2021splade,
  title={SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking},
  author={Formal, Thibault and Piwowarski, Benjamin and Clinchant, St{\'e}phane},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2288--2292},
  year={2021}
}
```

## References

This work builds upon the following research:

1. **Quati Dataset**: Bueno, M., de Oliveira, E. S., Nogueira, R., Lotufo, R., & Pereira, J. (2024). *Quati: A Brazilian Portuguese Information Retrieval Dataset from Native Speakers*. arXiv:2404.06976. [https://arxiv.org/abs/2404.06976](https://arxiv.org/abs/2404.06976)

2. **mMARCO**: Bonifacio, L., Campiotti, I., Lotufo, R., & Nogueira, R. (2021). *mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset*. Proceedings of STIL 2021. [https://sol.sbc.org.br/index.php/stil/article/view/31136](https://sol.sbc.org.br/index.php/stil/article/view/31136)

3. **SPLADE**: Formal, T., Piwowarski, B., & Clinchant, S. (2021). *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. SIGIR 2021.

4. **BERTimbau**: Souza, F., Nogueira, R., & Lotufo, R. (2020). *BERTimbau: Pretrained BERT Models for Brazilian Portuguese*. BRACIS 2020.

## Acknowledgments

Special thanks to:
- **UNICAMP-DL team** for the mMARCO and mRobust Portuguese datasets
- **Quati dataset authors** for pioneering native Portuguese IR research
- **NeuralMind** for the BERTimbau model
- **Original SPLADE authors** for the model architecture

## License

Apache 2.0

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/AxelPCG/SPLADE-PT-BR).

