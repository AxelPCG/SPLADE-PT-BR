# SPLADE-PT-BR: Complete Usage Guide

This guide demonstrates how to use the SPLADE-PT-BR model for information retrieval in Portuguese.

## 📋 Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Inverted Index Search System](#inverted-index-search-system)
- [Vector Database Integration](#vector-database-integration)
- [Complete RAG System](#complete-rag-system)
- [Optimizations](#optimizations)
- [Practical Examples](#practical-examples)

---

## 🔧 Installation

### Requirements

```bash
pip install torch transformers huggingface_hub
```

### Load the Model

```python
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade
import torch

# Load model from Hugging Face
# Note: SPLADE is a custom architecture that wraps a BERT-MLM model
# You cannot use AutoModel.from_pretrained() - must instantiate Splade class directly
model = Splade(
    model_type_or_dir="AxelPCG/splade-pt-br",  # HF repo with trained BERT-MLM weights
    agg="max"  # Aggregation method (max or sum) - use "max" for this model
)
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Set to evaluation mode and move to GPU (if available)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"✅ Model loaded on device: {device}")
```

> **⚠️ Common Error:** 
> - ❌ `model = AutoModel.from_pretrained("AxelPCG/splade-pt-br")` → This will fail!
> - ❌ `model = Splade.from_pretrained("AxelPCG/splade-pt-br")` → `from_pretrained()` doesn't exist on Splade class
> - ✅ `model = Splade(model_type_or_dir="AxelPCG/splade-pt-br", agg="max")` → Correct way!

---

## 💡 Basic Usage

### Encode Queries and Documents

```python
def encode_text(text, is_query=True):
    """
    Encodes text into SPLADE sparse vector
    
    Args:
        text: Text to encode
        is_query: True for queries, False for documents
    
    Returns:
        dict with 'indices' (non-zero positions) and 'values' (weights)
    """
    # Tokenize
    tokens = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=False
    )
    
    # Move to correct device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Encode
    with torch.no_grad():
        if is_query:
            vec = model(q_kwargs=tokens)["q_rep"].squeeze()
        else:
            vec = model(d_kwargs=tokens)["d_rep"].squeeze()
    
    # Extract sparse representation
    indices = torch.nonzero(vec).squeeze().cpu()
    if indices.dim() == 0:  # Single element case
        indices = indices.unsqueeze(0)
    indices = indices.tolist()
    
    values = vec[indices].cpu().tolist()
    
    return {
        "indices": indices,
        "values": values,
        "num_active": len(indices),
        "sparsity": 1 - (len(indices) / vec.shape[0])
    }

# Usage example (Portuguese queries - model trained for PT-BR)
query = "Qual é a capital do Brasil?"
doc = "Brasília é a capital federal do Brasil desde 1960."

query_vec = encode_text(query, is_query=True)
doc_vec = encode_text(doc, is_query=False)

print(f"Query - Active dimensions: {query_vec['num_active']}")
print(f"Query - Sparsity: {query_vec['sparsity']:.2%}")
print(f"Doc - Active dimensions: {doc_vec['num_active']}")
```

### Calculate Similarity

```python
def calculate_similarity(query_vec, doc_vec):
    """
    Calculate similarity between query and document (dot product)
    
    Args:
        query_vec, doc_vec: Outputs from encode_text()
    
    Returns:
        float: Similarity score
    """
    # Convert to dictionary for fast access
    doc_dict = dict(zip(doc_vec["indices"], doc_vec["values"]))
    
    # Dot product only on common dimensions
    score = sum(
        q_val * doc_dict[q_idx]
        for q_idx, q_val in zip(query_vec["indices"], query_vec["values"])
        if q_idx in doc_dict
    )
    
    return score

# Calculate similarity
similarity = calculate_similarity(query_vec, doc_vec)
print(f"Similarity: {similarity:.4f}")
```

### Inspect Expanded Terms

```python
def inspect_sparse_representation(sparse_vec, tokenizer, top_k=20):
    """
    Mostra os termos mais importantes na representação esparsa
    
    Args:
        sparse_vec: Saída de encode_text()
        tokenizer: Tokenizer do modelo
        top_k: Número de top termos para mostrar
    """
    # Ordenar por peso
    sorted_pairs = sorted(
        zip(sparse_vec["indices"], sparse_vec["values"]),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    # Convert indices to tokens
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    print(f"\nTop {top_k} termos:")
    for idx, weight in sorted_pairs:
        token = id_to_token.get(idx, f"<UNK_{idx}>")
        print(f"  {token:20s} → {weight:.4f}")

# Inspect query (Portuguese - model trained for PT-BR)
query = "remédio para dor de cabeça"
query_vec = encode_text(query, is_query=True)
inspect_sparse_representation(query_vec, tokenizer, top_k=15)

# You will see semantic expansions like:
# remédio          → 2.5000
# medicamento      → 2.1000  ← Expanded!
# analgésico       → 1.8000  ← Expanded!
# dor              → 2.3000
# cabeça           → 2.0000
# enxaqueca        → 1.5000  ← Expanded!
```

---

## 🗄️ Inverted Index Search System

### Basic Implementation

This implementation uses a simple inverted index that can be easily adapted for any vector database.

```python
class SimpleSparseRetriever:
    """Simple search system using inverted index"""
    
    def __init__(self):
        self.documents = {}
        self.inverted_index = {}
    
    def add_document(self, doc_id, text):
        """Add document to index"""
        # Store document
        self.documents[doc_id] = text
        
        # Encode document
        doc_vec = encode_text(text, is_query=False)
        
        # Add to inverted index
        for idx, value in zip(doc_vec["indices"], doc_vec["values"]):
            if idx not in self.inverted_index:
                self.inverted_index[idx] = []
            self.inverted_index[idx].append((doc_id, value))
    
    def add_documents_batch(self, documents):
        """Add multiple documents"""
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)
        print(f"✅ {len(documents)} documents indexed")
    
    def search(self, query, top_k=10):
        """
        Busca documentos relevantes
        
        Args:
            query: Texto da query
            top_k: Número de resultados
        
        Returns:
            Lista de (doc_id, text, score)
        """
        # Codificar query
        query_vec = encode_text(query, is_query=True)
        
        # Calculate scores using inverted index
        scores = {}
        for idx, q_value in zip(query_vec["indices"], query_vec["values"]):
            if idx in self.inverted_index:
                for doc_id, d_value in self.inverted_index[idx]:
                    scores[doc_id] = scores.get(doc_id, 0) + (q_value * d_value)
        
        # Sort by score
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(doc_id, self.documents[doc_id], score) for doc_id, score in results]
    
    def get_stats(self):
        """Return index statistics"""
        return {
            "num_documents": len(self.documents),
            "num_terms": len(self.inverted_index),
            "avg_doc_length": sum(len(self.inverted_index.get(i, [])) for i in range(len(self.inverted_index))) / max(1, len(self.documents))
        }

# Usage example (Portuguese content - model trained for PT-BR)
retriever = SimpleSparseRetriever()

# Index documents
docs = {
    1: "Brasília é a capital do Brasil desde 1960.",
    2: "O Python é uma linguagem de programação de alto nível.",
    3: "A Copa do Mundo de 2014 foi realizada no Brasil.",
    4: "Machine learning é um subcampo da inteligência artificial.",
    5: "São Paulo é a maior cidade do Brasil."
}

retriever.add_documents_batch(docs)

# Search
query = "Qual a capital brasileira?"
results = retriever.search(query, top_k=3)

print(f"\n🔍 Results for: '{query}'\n")
for i, (doc_id, text, score) in enumerate(results, 1):
    print(f"{i}. [Score: {score:.4f}] {text}")

# Statistics
stats = retriever.get_stats()
print(f"\n📊 Index statistics:")
print(f"   Documents: {stats['num_documents']}")
print(f"   Unique terms: {stats['num_terms']}")
```

---

## 🔌 Vector Database Integration

The output format of SPLADE-PT-BR (sparse vectors with indices and values) is compatible with any vector database that supports sparse vectors. Here is the generic format:

### Output Format

```python
# Each sparse vector is represented as:
sparse_vector = {
    "indices": [45, 128, 892, 1024, ...],  # Active token IDs (list of integers)
    "values": [2.5, 1.8, 3.2, 1.1, ...],    # Corresponding weights (list of floats)
    "num_active": 120,                      # Number of active dimensions
    "sparsity": 0.996                       # Percentage of zeros (~99.6%)
}
```

### Adaptation for Your Vector Database

```python
def adapt_for_vector_db(sparse_vec):
    """
    Adapt this format for your specific vector database.
    
    Common format examples:
    - Some DBs need dict: {idx: value, ...}
    - Others need tuples: [(idx, value), ...]
    - Others accept separate arrays: indices[], values[]
    """
    # Format 1: Dictionary (common in many DBs)
    dict_format = {
        str(idx): float(val) 
        for idx, val in zip(sparse_vec["indices"], sparse_vec["values"])
    }
    
    # Formato 2: Lista de tuplas
    tuple_format = [
        (int(idx), float(val)) 
        for idx, val in zip(sparse_vec["indices"], sparse_vec["values"])
    ]
    
    # Format 3: Separate arrays (NumPy/JSON)
    array_format = {
        "indices": sparse_vec["indices"],  # List of ints
        "values": sparse_vec["values"]     # List of floats
    }
    
    # Return the format your DB needs
    return array_format  # or dict_format, or tuple_format

# Usage example (Portuguese text - model trained for PT-BR)
doc_vec = encode_text("Seu documento em português aqui", is_query=False)
db_format = adapt_for_vector_db(doc_vec)

# Now use db_format with your vector database API
# your_vector_db.insert(id=1, vector=db_format, metadata={...})
```

### Generic Search

```python
def search_in_vector_db(query_text, your_db_client, collection_name, top_k=10):
    """
    Generic template for searching in any vector database
    
    Adapt the API calls for your specific database
    """
    # 1. Encode query
    query_vec = encode_text(query_text, is_query=True)
    
    # 2. Adapt format
    db_query = adapt_for_vector_db(query_vec)
    
    # 3. Search (adapt this line for your API)
    # results = your_db_client.search(
    #     collection=collection_name,
    #     query_vector=db_query,
    #     limit=top_k
    # )
    
    # 4. Process results (format varies by DB)
    # return results
    pass
```

---

## 🤖 Complete RAG System

### RAG Pipeline with SPLADE Retriever

```python
class RAGPipeline:
    """Complete RAG pipeline with SPLADE retriever"""
    
    def __init__(self, retriever, llm_function):
        """
        Args:
            retriever: Instance of SimpleSparseRetriever
            llm_function: Function that receives prompt and returns answer
        """
        self.retriever = retriever
        self.llm = llm_function
    
    def query(self, question, top_k=3, return_sources=True):
        """
        Process question using RAG
        
        Args:
            question: User question
            top_k: Number of documents for context
            return_sources: Whether to return sources
        
        Returns:
            dict with 'answer', 'sources', 'scores'
        """
        # 1. Retrieve relevant documents
        print(f"🔍 Searching documents for: '{question}'")
        results = self.retriever.search(question, top_k=top_k)
        
        if not results:
            return {
                "answer": "I couldn't find relevant documents to answer your question.",
                "sources": [],
                "scores": []
            }
        
        # 2. Build context
        context_parts = []
        for i, (doc_id, text, score) in enumerate(results, 1):
            context_parts.append(f"[Document {i}] {text}")
        
        context = "\n\n".join(context_parts)
        
        # 3. Criar prompt
        prompt = f"""Baseado nos seguintes documentos, responda a pergunta de forma precisa e concisa.

Documents:
{context}

Question: {question}

Answer (based only on the provided documents):"""
        
        # 4. Generate answer with LLM
        print("🤖 Generating answer...")
        answer = self.llm(prompt)
        
        result = {
            "answer": answer,
            "sources": [text for _, text, _ in results],
            "scores": [score for _, _, score in results]
        }
        
        return result
    
    def format_response(self, result):
        """Format response for display"""
        output = f"\n📝 Answer:\n{result['answer']}\n"
        
        if result['sources']:
            output += f"\n📚 Sources ({len(result['sources'])}):\n"
            for i, (source, score) in enumerate(zip(result['sources'], result['scores']), 1):
                output += f"  {i}. [Score: {score:.3f}] {source}\n"
        
        return output

# Usage example with mock LLM (replace with your implementation)
def mock_llm(prompt):
    """
    Replace this function with your real LLM call
    (OpenAI, Anthropic, local model, etc.)
    """
    # Simple example: extract answer from context
    if "capital" in prompt.lower() and "brasil" in prompt.lower():
        return "Brasília é a capital do Brasil desde 1960."
    return "Com base nos documentos fornecidos, posso responder que..."

# Create RAG pipeline
rag = RAGPipeline(retriever, mock_llm)

# Ask question (Portuguese - model trained for PT-BR)
question = "Qual é a capital do Brasil?"
result = rag.query(question, top_k=3)
print(rag.format_response(result))
```

---

## ⚡ Optimizations

### Batch Encoding

```python
def encode_batch(texts, is_query=True, batch_size=32):
    """
    Encode multiple texts in batch for better efficiency
    
    Args:
        texts: List of texts
        is_query: True for queries, False for documents
        batch_size: Batch size
    
    Returns:
        List of dicts with sparse representations
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Tokenizar batch
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Codificar
        with torch.no_grad():
            if is_query:
                vecs = model(q_kwargs=tokens)["q_rep"]
            else:
                vecs = model(d_kwargs=tokens)["d_rep"]
        
        # Processar cada vetor
        for vec in vecs:
            indices = torch.nonzero(vec).squeeze().cpu()
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
            indices = indices.tolist()
            values = vec[indices].cpu().tolist()
            
            results.append({
                "indices": indices,
                "values": values,
                "num_active": len(indices)
            })
    
    return results

# Example: encode 100 documents in batch (Portuguese - model trained for PT-BR)
docs = [f"Documento sobre tópico {i}" for i in range(100)]
encoded = encode_batch(docs, is_query=False, batch_size=16)
print(f"✅ {len(encoded)} documents encoded")
```

### Embedding Cache

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def encode_cached(text, is_query=True):
    """Cached version of encode_text"""
    result = encode_text(text, is_query=is_query)
    # Convert lists to tuples to be hashable
    return tuple(result["indices"]), tuple(result["values"])

# Usage: automatic caching for repeated texts (Portuguese queries)
vec1 = encode_cached("capital do Brasil", is_query=True)  # Calculates
vec2 = encode_cached("capital do Brasil", is_query=True)  # Returns from cache
```

---

## 📚 Practical Examples

### Example 1: FAQ Search

```python
# Create retriever
faq_retriever = SimpleSparseRetriever()

# Add FAQs (Portuguese content - model trained for PT-BR)
faqs = {
    1: "Como faço para resetar minha senha? Acesse a página de login e clique em 'Esqueci minha senha'.",
    2: "Qual o prazo de entrega? O prazo é de 5-7 dias úteis para todo o Brasil.",
    3: "Aceitam cartão de crédito? Sim, aceitamos Visa, Mastercard e Elo.",
    4: "Como faço para cancelar meu pedido? Entre em contato com o suporte até 24h após a compra.",
    5: "Tem desconto para estudantes? Sim, 15% de desconto com comprovante de matrícula."
}

faq_retriever.add_documents_batch(faqs)

# Search for similar FAQ
pergunta_usuario = "esqueci minha senha como recuperar"
results = faq_retriever.search(pergunta_usuario, top_k=1)

if results:
    doc_id, faq_text, score = results[0]
    print(f"Matching FAQ (score: {score:.2f}):")
    print(f"  {faq_text}")
```

### Example 2: Knowledge Base Search

```python
# Technical knowledge base
kb_retriever = SimpleSparseRetriever()

# Articles in Portuguese (model trained for PT-BR)
articles = {
    1: "Python é uma linguagem interpretada de alto nível, conhecida por sua sintaxe clara.",
    2: "Machine Learning é um subcampo da IA que permite sistemas aprenderem com dados.",
    3: "Deep Learning usa redes neurais profundas para resolver problemas complexos.",
    4: "Natural Language Processing permite computadores entenderem linguagem humana.",
    5: "Computer Vision permite máquinas interpretarem e entenderem imagens."
}

kb_retriever.add_documents_batch(articles)

# Search with semantic expansion (Portuguese queries)
queries = [
    "aprendizado de máquina",  # Will find "Machine Learning"
    "processamento de texto",  # Will find "Natural Language Processing"
    "linguagem de programação" # Will find "Python"
]

for query in queries:
    print(f"\n🔍 Query: {query}")
    results = kb_retriever.search(query, top_k=2)
    for i, (_, text, score) in enumerate(results, 1):
        print(f"  {i}. [Score: {score:.2f}] {text[:80]}...")
```

### Example 3: Content Recommendation System

```python
class ContentRecommender:
    """SPLADE-based content recommender"""
    
    def __init__(self):
        self.retriever = SimpleSparseRetriever()
        self.user_history = {}
    
    def add_content(self, content_id, title, description):
        """Add content to system"""
        full_text = f"{title}. {description}"
        self.retriever.add_document(content_id, full_text)
    
    def record_interaction(self, user_id, content_id):
        """Record user interaction"""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id].append(content_id)
    
    def recommend(self, user_id, top_k=5):
        """Recommend content based on history"""
        if user_id not in self.user_history or not self.user_history[user_id]:
            return []
        
        # Use last items from history as "query"
        recent_items = self.user_history[user_id][-3:]
        query_texts = [self.retriever.documents[item_id] for item_id in recent_items if item_id in self.retriever.documents]
        
        if not query_texts:
            return []
        
        # Combine recent texts as query
        combined_query = " ".join(query_texts)
        
        # Search for similar content
        results = self.retriever.search(combined_query, top_k=top_k*2)
        
        # Filter already seen items
        recommendations = [
            (doc_id, text, score)
            for doc_id, text, score in results
            if doc_id not in self.user_history[user_id]
        ][:top_k]
        
        return recommendations

# Usage example (Portuguese content - model trained for PT-BR)
recommender = ContentRecommender()

# Add content
contents = {
    1: ("Python Básico", "Aprenda os fundamentos da linguagem Python"),
    2: ("Machine Learning", "Introdução ao aprendizado de máquina com Python"),
    3: ("Deep Learning", "Redes neurais e deep learning na prática"),
    4: ("Web Scraping", "Colete dados da web com Python"),
    5: ("Data Science", "Análise de dados com pandas e numpy")
}

for content_id, (title, desc) in contents.items():
    recommender.add_content(content_id, title, desc)

# Simulate interactions
recommender.record_interaction("user1", 1)  # Viewed Python Básico
recommender.record_interaction("user1", 2)  # Viewed Machine Learning

# Recommend
recommendations = recommender.recommend("user1", top_k=3)
print("\n💡 Recommendations for user1:")
for i, (content_id, text, score) in enumerate(recommendations, 1):
    print(f"  {i}. [Score: {score:.2f}] {text[:60]}...")
```

---

## 🎯 Performance Tips

1. **Use GPU**: 5-10x mais rápido que CPU para encoding
2. **Batch Processing**: Processe múltiplos textos de uma vez
3. **Cache**: Use cache para queries/documentos repetidos
4. **Preprocessing**: Remova HTML, normalize texto antes de codificar
5. **Índice Otimizado**: Para grandes coleções, considere índices persistentes

---

## 🐛 Troubleshooting

### Error: "model type `splade` not recognized by Transformers"

**Problem:** Você tentou usar `AutoModel.from_pretrained()` ou `AutoTokenizer.from_pretrained()` com o modelo SPLADE.

**Cause:** SPLADE é uma arquitetura customizada que não está registrada no registro do Transformers.

**Solução:**

```python
# ❌ ERRADO - Não funciona
from transformers import AutoModel
model = AutoModel.from_pretrained("AxelPCG/splade-pt-br")  # Erro!

# ✅ CORRECT - Use the Splade class directly
from splade.models.transformer_rep import Splade
model = Splade(
    model_type_or_dir="AxelPCG/splade-pt-br",
    agg="max"
)

# Para o tokenizer, use o modelo base (BERTimbau)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
```

**Por que isso acontece?**
- O repo HF contém apenas os pesos do BERT-MLM treinado
- A lógica SPLADE (agregação max, sparse encoding) está na classe Python `Splade`
- Você precisa instanciar a classe manualmente passando o caminho do HF

### Erro: Out of Memory

```python
# Reduce batch size
encoded = encode_batch(texts, batch_size=8)  # instead of 32
```

### Baixa Qualidade nos Resultados

```python
# Verificar sparsity
vec = encode_text(query, is_query=True)
print(f"Sparsity: {vec['sparsity']:.2%}")
# Ideal: > 99%

# Se muito denso (< 95%), verifique:
# 1. Modelo carregado corretamente
# 2. Text not too long (> 256 tokens)
# 3. Tokenizer correto (neuralmind/bert-base-portuguese-cased)
```

---

---

## 🔧 Integration with Popular Frameworks

### Generic Template

```python
class VectorDBAdapter:
    """
    Generic adapter class for any vector database
    Implement the abstract methods for your specific DB
    """
    
    def __init__(self, db_client):
        self.client = db_client
    
    def format_vector(self, sparse_vec):
        """Convert to your DB format"""
        raise NotImplementedError("Implement for your DB")
    
    def insert_document(self, doc_id, text, metadata=None):
        """Insert document into DB"""
        vec = encode_text(text, is_query=False)
        db_vec = self.format_vector(vec)
        # self.client.insert(doc_id, db_vec, metadata)
        raise NotImplementedError("Implement for your DB")
    
    def search(self, query, top_k=10, filters=None):
        """Search documents"""
        query_vec = encode_text(query, is_query=True)
        db_query = self.format_vector(query_vec)
        # results = self.client.search(db_query, top_k, filters)
        # return results
        raise NotImplementedError("Implement for your DB")

# Usage (example in Portuguese, as the model is trained for PT-BR):
# adapter = VectorDBAdapter(your_db_client)
# adapter.insert_document(1, "Texto do documento", {"categoria": "tech"})
# results = adapter.search("your query in Portuguese", top_k=5)
```

### Example: Format for Different DBs

```python
# For DBs that use string dictionaries
def format_as_string_dict(sparse_vec):
    return {str(idx): float(val) for idx, val in zip(sparse_vec["indices"], sparse_vec["values"])}

# For DBs that use NumPy arrays
def format_as_numpy(sparse_vec):
    import numpy as np
    dense = np.zeros(29794)  # Vocabulary size
    for idx, val in zip(sparse_vec["indices"], sparse_vec["values"]):
        dense[idx] = val
    return dense

# For DBs that use COO (Coordinate) format
def format_as_coo(sparse_vec):
    return {
        "indices": sparse_vec["indices"],
        "values": sparse_vec["values"],
        "shape": (29794,)
    }
```

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/AxelPCG/SPLADE-PT-BR/issues)
- **Model Card**: [Hugging Face](https://huggingface.co/AxelPCG/splade-pt-br)

---

**Happy Searching! 🚀**
