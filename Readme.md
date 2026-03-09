# GraphRAG From Scratch

A lightweight Retrieval-Augmented Generation (RAG) pipeline that builds a Knowledge Graph from raw text using LLM-based triplet extraction, retrieves relevant subgraphs using semantic similarity, and generates answers grounded in the graph.

The goal of this project is to implement a GraphRAG system from scratch, without relying on existing graph-RAG frameworks.

The pipeline extracts structured knowledge from text, builds a graph representation, retrieves relevant facts using graph traversal + semantic filtering, and uses them to generate answers.

---

# Overview

Traditional RAG systems retrieve documents or passages.  
GraphRAG instead retrieves structured knowledge from a graph.

This project implements the following pipeline:

```
Raw Text
   │
   ▼
LLM Triplet Extraction
   │
   ▼
Knowledge Graph Construction
   │
   ▼
Graph Retrieval 
   │
   ▼
Context Filtering
   │
   ▼
LLM Answer Generation
```

The system combines:

- LLMs for knowledge extraction
- NetworkX for graph representation
- SentenceTransformers for semantic retrieval

---

# Features

- Triplet extraction using an LLM
- Automatic Knowledge Graph construction
- Graph traversal retrieval (BFS hops)
- Semantic filtering using embeddings
- Local or remote LLM answering
- Dataset evaluation using SQuAD
- Modular architecture for experimentation

---

# Project Structure

```
GraphRAG/
│
├── main.py
├── evaluation.py
│
├── src/
│   ├── extractor.py
│   ├── graph_builder.py
│   ├── retriever.py
│   └── answerer.py
│
├── data/
│   └── raw_text_data/
│
├── model_instructions/
│   ├── new_extraction_instruction.txt
│   └── answer_instruction.txt
│
├── outputs/
│   └── raw_triplets.txt
│
└── README.md
```

---

# System Components

## 1. Triplet Extraction

`src/extractor.py`

Extracts structured knowledge triplets from text using an LLM.

Example output:

```
("Albert Einstein", "born in", "Ulm")
("Albert Einstein", "developed", "Theory of Relativity")
("Ulm", "located in", "Germany")
```

These triplets form the basis of the knowledge graph.

The extraction is chunked to avoid exceeding the LLM context window.

---

## 2. Knowledge Graph Construction

`src/graph_builder.py`

Triplets are converted into a directed graph using NetworkX.

```
Albert Einstein ── born in ──► Ulm
Ulm ── located in ──► Germany
```

Duplicate and invalid edges are automatically filtered.

---

## 3. Graph Retrieval

`src/retriever.py`

Retrieval is performed in two steps.

### Step 1 — Semantic Node Retrieval

The most relevant graph nodes are identified using embeddings.

Example:

```
Query: "Where was Einstein born?"

→ Most relevant node: "Albert Einstein"
```

### Step 2 — Graph Traversal

Breadth-First Search explores the graph neighborhood.

```
Albert Einstein → Ulm → Germany
```

This produces candidate triplets.

---

## 4. Context Filtering

Triplets retrieved through traversal are filtered using semantic similarity.

This step removes irrelevant graph edges.

---

## 5. Answer Generation

`src/answerer.py`

Relevant triplets are converted to natural language context and passed to the answering LLM.

Example context:

```
Albert Einstein born in Ulm
Ulm located in Germany
```

The LLM generates a final answer grounded in the retrieved facts.

---

# Running the System

## 1. Install Dependencies

```
pip install -r requirements.txt
```

Recommended libraries:

```
networkx
sentence-transformers
datasets
python-dotenv
smolagents
ollama
```

---

## 2. Prepare Text Data

Place `.txt` files inside:

```
data/raw_text_data/
```

Example:

```
Albert_Einstein.txt
World_War_II.txt
Physics.txt
```

---

## 3. Run the Pipeline

```
python main.py
```

Pipeline steps:

1. Merge text documents
2. Extract triplets
3. Build knowledge graph
4. Retrieve relevant facts
5. Generate answer

Example query:

```
Where did Albert Einstein live?
```

Example output:

```
Retrieved Facts:
- Albert Einstein born in Ulm
- Ulm located in Germany

Answer:
Albert Einstein was born in Ulm, Germany.
```

---

# Evaluation

The system includes evaluation using the SQuAD dataset.

Evaluation protocol:

- Use the first 100 questions
- Extract 20 shared contexts
- Build the knowledge graph from those contexts
- Evaluate question answering performance

Metrics used:

- Exact Match (EM)
- F1 Score

Run evaluation:

`
python evaluation.py
```

Example output:

```
Example 1
Q: Where was Einstein born?
Gold: Ulm
Pred: Ulm
EM: 1
F1: 1.0
```

Final metrics:

```
Exact Match: ???
F1 Score: ???
```

---

# Example Graph

Example knowledge graph extracted from text:

```
Albert Einstein
   │
   ├── born in → Ulm
   │
   ├── developed → Theory of Relativity
   │
   └── won → Nobel Prize
```

Graph retrieval allows multi-hop reasoning across entities.

---

# Models Used

### Extraction Model

```
Qwen2.5-Coder-7B-Instruct
```

Used for knowledge triplet extraction.

### Answering Model

```
Meta-Llama-3.1-8B-Instruct
```

Used for final answer generation.

### Retrieval Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

Used for semantic similarity and retrieval.

---

# Design Goals

This project focuses on:

- Building a GraphRAG pipeline from scratch
- Understanding knowledge graph retrieval
- Experimenting with LLM-based information extraction
- Evaluating graph-based RAG architectures

The implementation prioritizes clarity and modularity to make experimentation easy.

---

# Future Improvements

Potential extensions include:

- Graph-based reasoning chains
- Relation normalization
- Entity linking
- Graph embedding retrieval
- Hybrid document + graph retrieval
- Graph pruning
- Multi-hop reasoning benchmarks

---
