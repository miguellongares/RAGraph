import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph:
    def __init__(self):
        #It is important to use a directed graph (DiGraph) Albert -> Expert -> Phisicist is different from Physicist -> Expert -> Albert
        self.G = nx.DiGraph()

    def add_triplet(self, subject, relation, object):
        #Skip adding duplicate triplets:
        if self.G.has_edge(subject, object):
            if self.G[subject][object]['relation'] == relation:
                return # Skip adding this triplet as it already exists
        #Skip invalid triplets with same subject and object:
        if subject == object:
            return # Skip adding this triplet as it is invalid   
        self.G.add_edge(subject, object, relation=relation)

    def build_graph(self, triplets):
        for triplet in triplets:
            subject, relation, object = triplet
            self.add_triplet(subject, relation, object)
        return self.G
    
    def display_graph(self):
        nx.draw(self.G, with_labels=True)    
        plt.show()


#Create a dummy function to return a knowledge graph from a list of triplets:
def create_dummy_knowledge_graph():
    list_of_triplets = [
        ("Albert Einstein", "developed", "Theory of Relativity"),
        ("Albert Einstein", "born in", "Ulm, Germany"),
        ("Albert Einstein", "died in", "Princeton, USA"),
        ("Theory of Relativity", "published in", "1905"),
        ('Hitler', 'born in', 'Braunau am Inn, Austria'),
        ('Hitler', 'perscuted', 'Jews'),
        ('Albert Einstein', 'was', 'Jews'),
    ]
    KG = KnowledgeGraph()
    KG.build_graph(list_of_triplets)
    return KG


if __name__ == "__main__":
    #Example usage
    list_of_triplets = [
        ("Albert Einstein", "developed", "Theory of Relativity"),
        ("Albert Einstein", "born in", "Ulm, Germany"),
        ("Albert Einstein", "died in", "Princeton, USA"),
        ("Theory of Relativity", "published in", "1905"),
        ('Hitler', 'born in', 'Braunau am Inn, Austria'),
        ('Hitler', 'perscuted', 'Jews'),
        ('Albert Einstein', 'was', 'Jews'),
    ]
    KG = KnowledgeGraph()
    KG.build_graph(list_of_triplets)
    KG.display_graph()