import numpy as np
from collections import deque
from sentence_transformers import SentenceTransformer
from transformers import logging

# Suppress unnecessary transformer warnings
logging.set_verbosity_error()

# Global default model with local caching
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class GraphRetriever:
    def __init__(self, knowledge_graph, model_name=DEFAULT_MODEL):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = SentenceTransformer(model_name, cache_folder="./models")
        
        # Pre-calculate node embeddings to make retrieval near-instant
        self.nodes_list = list(self.knowledge_graph.G.nodes)
        if self.nodes_list:
            self.node_embeddings = self.embedding_model.encode(self.nodes_list)
        else:
            self.node_embeddings = None

    def _get_relevant_nodes(self, query, top_k=2):
        """Finds the 'entry points' into the graph based on semantic similarity."""
        if not self.nodes_list:
            return []

        query_embedding = self.embedding_model.encode(query)
        similarities = self.embedding_model.similarity(query_embedding, self.node_embeddings)[0]
        
        # Get indices of top_k most similar nodes
        top_k_idx = np.argsort(similarities)[-top_k:]
        return [self.nodes_list[idx] for idx in top_k_idx]

    def retrive_triplets_from_knowledgegraph(self, query, top_k=2, hops=2):
        """
        Performs a Breadth-First Search (BFS) starting from the most relevant nodes.
        Returns a list of triplets (subject, relation, object).
        """
        seed_nodes = self._get_relevant_nodes(query, top_k)
        
        visited = set(seed_nodes)
        queue = deque([(node, 0) for node in seed_nodes])
        collected_triplets = []

        while queue:
            current_node, current_depth = queue.popleft()

            if current_depth >= hops:
                continue

            # Iterate through neighbors in the NetworkX graph
            for neighbor in self.knowledge_graph.G[current_node]:
                # Extract edge attributes
                edge_data = self.knowledge_graph.G[current_node][neighbor]
                relation = edge_data.get('relation', 'related_to')
                
                collected_triplets.append((current_node, relation, neighbor))

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))

        return list(set(collected_triplets))  # Remove duplicates if any

    def filter_relevant_triplets(self, query, triplets, threshold=0.4):
        """Filters the graph-traversed triplets using a semantic similarity threshold."""
        if not triplets:
            return []

        # Convert tuples to strings for embedding: ('Einstein', 'born in', 'Germany') -> 'Einstein born in Germany'
        triplet_strings = [' '.join(map(str, t)) for t in triplets]
        
        query_emb = self.embedding_model.encode(query)
        triplet_embs = self.embedding_model.encode(triplet_strings)
        
        similarities = self.embedding_model.similarity(query_emb, triplet_embs)[0]

        return [
            triplets[i] for i, score in enumerate(similarities) 
            if score >= threshold
        ]

# --- SELF-TEST BLOCK ---
if __name__ == "__main__":
    # This block only runs if you execute retriever.py directly
    from src.graph_builder import KnowledgeGraph 
    
    # Mock setup
    kg = KnowledgeGraph()
    kg.build_graph([("Albert Einstein", "born in", "Ulm"), ("Ulm", "located in", "Germany")])
    
    retriever = GraphRetriever(kg)
    test_query = "Where was Einstein born?"
    
    results = retriever.retrive_triplets_from_knowledgegraph(test_query, top_k=1, hops=2)
    filtered = retriever.filter_relevant_triplets(test_query, results, threshold=0.3)
    
    print(f"Query: {test_query}")
    print(f"Retrieved: {filtered}")