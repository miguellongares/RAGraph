# Proof of Concept: GraphRAG System

This repository contains a lightweight **Graph Retrieval-Augmented Generation (GraphRAG)** pipeline designed to extract structured knowledge from text and provide context-aware answers.

## Architecture & Tech Stack

* **Information Extraction:** An LLM agent (≤ 8B parameters) using specialized instruction prompts to extract entity-relation-entity triplets.
* **Knowledge Graph:** Built using **NetworkX** to manage and store extracted triplets as a directed graph.
* **Vector Embeddings:** **SentenceTransformers** (`all-MiniLM-L6-v2`) for entity similarity mapping and retrieving relevant graph sub-structures.
* **Response Generation:** A final LLM stage that synthesizes a user answer based on retrieved graph context.
* **Implementation:** Pure Python.

---

## Future Improvements

### 1. Triplet Extraction Quality
Currently, triplet retrieval accuracy is the primary area for growth. Planned optimizations include:
* **Scaling:** Testing larger LLMs (e.g., 14B or 32B parameters).
* **Prompt Engineering:** Developing more robust Few-Shot templates with clearer examples.
* **Multi-Step Pipeline:** Implementing a "Decomposition" strategy: first identifying all entities, then extracting relationships between them.
* **Self-Correction:** Using a "Critic" agent to validate and prune extracted triplets.
* **Agentic Logic:** Evaluating if a full ReAct agent loop outperforms basic zero-shot generation.

### 2. Graph Refinement
* **Entity Resolution:** Implementing a similarity model to detect and **merge duplicate or synonymous nodes** (e.g., merging "Albert Einstein" and "A. Einstein").
* **Edge Weighting:** Assigning weights to relationships based on extraction confidence or frequency.

---

## Technical Notes
Initially, the `transformers` library was used for local model management. However, this proved **inefficient** for rapid prototyping. The system now utilizes **Ollama** and **smolagents** to handle model inference, significantly improving the speed of text processing and triplet extraction.