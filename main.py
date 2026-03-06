
import time 
from smolagents import InferenceClientModel
from dotenv import load_dotenv
import os

from extractor import ExtractionModel
from graph_build import KnowledgeGraph
from graph_retriver import GraphRetriver
from answerer import AnswererModel

###Use model form Hugging Face Inference API:
#load the HF_Token from the .env file
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
model_for_extraction = InferenceClientModel(model_id = 'Qwen/Qwen2.5-Coder-7B-Instruct')
model_for_answer = InferenceClientModel(model_id = 'Qwen/Qwen2.5-Coder-7B-Instruct')

#>Initialize the extraction model
extraction_model = ExtractionModel(model_for_extraction)
#>Initialize the knowledge graph
KG = KnowledgeGraph()
#>Initialize the answering model
answer_model = AnswererModel(model_for_answer)

#Execute printing process steps or without:
verbose = False


if __name__ == "__main__":
#Load document text to process:
    with open("Data/Albert_Einstein.txt", "r") as f:
        text = f.read()
    
#Creation of infromation triplets from all the wanted documents:
    start_time = time.time() #Timer to meassure extraction time
    
    #Check if the extraction has been already made, since it is the most demanding task:
    triplets_path = 'Outputs/raw_triplets.txt'
    if os.path.exists(triplets_path):
        with open(triplets_path, 'r') as f:
            raw_triplets = f.read()
        triplets = extraction_model.get_triplets(raw_triplets)
    #If no extraction made, extract triplets and store it in the triplets_path
    else:
        with open("Data/Albert_Einstein.txt", "r") as f:
            text = f.read()
        #Extraction model crates triplets in raw text
        raw_triplets = extraction_model.generate(text)
        #Store the raw_triplets in the triplets_path:
        with open(triplets_path, 'w') as f:
            f.write(raw_triplets)
        #Extracting triplets from the model output...
        print(f'Raw triplets saved to {triplets_path}')
        triplets = extraction_model.get_triplets(raw_triplets)

    end_time = time.time()#End the timer
    if verbose: print(f"Time taken for text extraction: {end_time - start_time} seconds")

#Create the knowledge graph:
    start_time = time.time() #Timer for knowledge graph creation
    KG.build_graph(triplets)
    end_time = time.time()
    if verbose: print(f"Time taken for graph construction: {end_time - start_time} seconds")

#Graph retriever:
    start_time = time.time() #Timer for graph retriver initialization
    retriever = GraphRetriver(KG)
    end_time = time.time()
    if verbose: print(f"Time taken for graph retriever initialization: {end_time - start_time} seconds")

#Retrive from graph the most relevant infromation based on the query:
    start_time = time.time()
    query = "What was the relation between Albert Einstein and Hitler?"
    top_k_triplets = retriever.retrieve(query)

    #Print the retrieved triplets for the query:
    if verbose: print(f"Retrieved triplets for the query: {query}")
    if verbose: print(top_k_triplets)
    end_time = time.time()
    if verbose: print(f"Time taken for graph retrieval: {end_time - start_time} seconds")

#Use the answer model to respond to the query using the retrieved information:
    print('\n\n')
    print(f'The answering the question:\n{query}')
    answer_model.generate(query=query, triplets=top_k_triplets)
    
