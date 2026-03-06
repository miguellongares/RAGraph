# Open to discussion is which embedding model to use. The one used in the notebook is 'all-MiniLM-L6-v2'
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import logging
logging.set_verbosity_error()

#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./models"
)

class GraphRetriver:
    def __init__(self, knowledge_graph, embedding_model = embedding_model):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model
        #Generate the embeddings for the knowledge graph nodes to speed up the retrieval process
        self.node_embeddings = self.embedding_model.encode(list(self.knowledge_graph.G.nodes))
    
    def retrieve(self, query, top_k=2):
        #Embed the query
        query_embedding = self.embedding_model.encode(query)
        #Calculate the similarity between the query embedding and the node embeddings
        similarities = self.embedding_model.similarity(query_embedding, self.node_embeddings)
        #Get the top_k most similar nodes
        top_k_node_idx = np.argsort(similarities)[0][-top_k:]
        top_k_nodes = [list(self.knowledge_graph.G.nodes)[idx] for idx in top_k_node_idx]
        list_of_top_k_triplets = []
        for node in top_k_nodes:
            for neighbor in self.knowledge_graph.G.neighbors(node):
                (subject, relation, object) = (node,
                                               self.knowledge_graph.G[node][neighbor]['relation'],
                                               neighbor)
                list_of_top_k_triplets.append((subject, relation, object))

        return list_of_top_k_triplets
    

if __name__ == "__main__":
    from graph_build import create_dummy_knowledge_graph
    kg = create_dummy_knowledge_graph()
    retriever = GraphRetriver(kg)
    query = "Where was Albert Einstein born?"
    top_k_triplets = retriever.retrieve(query)
    print(top_k_triplets)