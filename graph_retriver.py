# Open to discussion is which embedding model to use. The one used in the notebook is 'all-MiniLM-L6-v2'
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import deque
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

    def get_relevant_nodes(self, query, top_k=2):
        #Embed the query
        query_embedding = self.embedding_model.encode(query)
        #Calculate the similarity between the query embedding and the node embeddings
        similarities = self.embedding_model.similarity(query_embedding, self.node_embeddings)
        #Get the top_k most similar nodes
        top_k_node_idx = np.argsort(similarities)[0][-top_k:]
        top_k_nodes = [list(self.knowledge_graph.G.nodes)[idx] for idx in top_k_node_idx]
        return top_k_nodes #list of the most relevat node names ['Albert', 'Phisics']

    def retrive_triplets_from_knowledgegraph(self, query, top_k=2, hops=2):
        #Get the relevant nodes
        relevant_nodes = self.get_relevant_nodes(query, top_k)
        #Create a set of visited nodes, a queue of nodes to visit tracking depth
        visited_nodes = set()
        nodes_to_visit = deque()
        list_of_triplets = []

        for node in relevant_nodes:
            visited_nodes.add(node)
            nodes_to_visit.append((node, 0)) # 0 sets the initial depth

        while nodes_to_visit:
            node, depth = nodes_to_visit.popleft()
            #Checks that the depth doesnt excede the max hops
            if depth >= hops:
                continue
            #Get all the triplets from the neighbors
            for neighbor in self.knowledge_graph.G[node]:
                #get subject, releation, object
                (s, r, o) = (node,
                             self.knowledge_graph.G[node][neighbor]['relation'],
                             neighbor)
                list_of_triplets.append((s,r,o))
                # Adds nodes to visit to the queue making sure that it has not been already visited
                if neighbor not in visited_nodes:
                    nodes_to_visit.append((neighbor, depth+1))
                    visited_nodes.add(neighbor)
            #End of retreve loop
        return list_of_triplets
    
    def filter_relevant_triplets(self, query, list_of_triplets, filter_portion=0.3):
        n_triplets_to_keep = int(len(list_of_triplets)*filter_portion)
        query_embedding = self.embedding_model.encode(query)
        #Convert each triplet into single text to encode the entire triplet:
        triplets_txt = [''.join(triplet) for triplet in list_of_triplets]
        triplets_embedding = self.embedding_model.encode(triplets_txt)
        similarities = self.embedding_model.similarity(query_embedding, triplets_embedding)
        top_triplets_idx = np.argsort(similarities)[0][-n_triplets_to_keep:]
        list_of_filered_triplets = [list_of_triplets[idx] for idx in top_triplets_idx]
        return list_of_filered_triplets

if __name__ == "__main__":
    from graph_build import create_dummy_knowledge_graph
    kg = create_dummy_knowledge_graph()
    retriever = GraphRetriver(kg)
    query = "Where was Albert Einstein born?"
    top_k_triplets = retriever.retrive_triplets_from_knowledgegraph(query)
    filterd_triplets = retriever.filter_relevant_triplets(query, top_k_triplets)
    for triplet in top_k_triplets: print(triplet)
    print('filerd ones: 0.5')
    for triplet in filterd_triplets: print(triplet)