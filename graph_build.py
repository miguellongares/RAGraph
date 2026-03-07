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
    ("Albert Einstein", "born on", "14 March 1879"),
    ("Albert Einstein", "died on", "18 April 1955"),
    ("Albert Einstein", "nationality", "German-born"),
    ("Albert Einstein", "known for", "developing the theory of relativity"),
    ("Albert Einstein", "contributed to", "quantum theory"),
    ("Albert Einstein", "mass-energy equivalence formula", "E = mc^2"),
    ("Albert Einstein", "received Nobel Prize in Physics in", "1921"),
    ("Albert Einstein", "reason for Nobel Prize", "his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect"),
    ("Albert Einstein", "moved to", "Switzerland in 1895"),
    ("Albert Einstein", "forsook German citizenship", "in 1896"),
    ("Albert Einstein", "enrolled in", "mathematics and physics teaching diploma program"),
    ("Albert Einstein", "program location", "Swiss federal polytechnic school in Zurich"),
    ("Albert Einstein", "graduated from", "University of Zurich in 1900"),
    ("Albert Einstein", "acquired Swiss citizenship", "in 1901"),
    ("Albert Einstein", "obtained citizenship", "and kept it for the rest of his life"),
    ("Albert Einstein", "secured permanent position at", "Swiss Patent Office in Bern"),
    ("Albert Einstein", "moved to Berlin in", "1914"),
    ("Albert Einstein", "joined Prussian Academy of Sciences", "in 1914"),
    ("Albert Einstein", "joined Humboldt University of Berlin", "in 1914"),
    ("Albert Einstein", "became director of", "Kaiser Wilhelm Institute for Physics in 1917"),
    ("Albert Einstein", "became a German citizen again", "in 1917"),
    ("Adolf Hitler", "came to power in", "Germany in 1933"),
    ("Albert Einstein", "visiting", "United States"),
    ("Albert Einstein", "decided to", "remain in the US due to Nazi persecution"),
    ("Albert Einstein", "granted", "American citizenship in 1940"),
    ("World War II", "was approaching", "and Albert Einstein warned President Franklin D. Roosevelt"),
    ("Albert Einstein", "endorsed letter to", "President Franklin D. Roosevelt"),
    ("Albert Einstein", " alerted President", "about German nuclear weapons program"),
    ("Albert Einstein", "recommended", "US begin similar research")
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