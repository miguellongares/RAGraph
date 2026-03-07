
import time 
from smolagents import InferenceClientModel
from dotenv import load_dotenv
import os

from extractor import ExtractionModel, merge_text_files
from graph_build import KnowledgeGraph
from graph_retriver import GraphRetriver
from answerer import AnswererModel

###Use model form Hugging Face Inference API:
#load the HF_Token from the .env file
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
model_for_extraction = InferenceClientModel(model_id = 'Qwen/Qwen2.5-Coder-7B-Instruct')
model_for_answer = InferenceClientModel(model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct', provider= 'novita')

#>Initialize the extraction model
extraction_model = ExtractionModel(model_for_extraction)
#>Initialize the knowledge graph
KG = KnowledgeGraph()
#>Initialize the answering model
answer_model = AnswererModel(model_for_answer)

#Execute printing process steps or without:
verbose = True


if __name__ == "__main__":
#Load document text to process:
    raw_text_data_folder= "Data/Raw_text_data"
    merged_text_file = "Data/all_text.txt"
    merge_text_files(
        data_folder=raw_text_data_folder,
        output_filename=merged_text_file)#All text in one txt file Data/all_text.txt
    
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
        #Extraction model crates triplets in raw text
        #Store the raw_triplets in the triplets_path:
        extraction_model.write_triplets(input_path= merged_text_file,
                                        output_path= triplets_path)
        #Extracting triplets from the model output...
        print(f'Raw triplets saved to {triplets_path}')

    end_time = time.time()#End the timer
    if verbose: print(f"Time taken for text extraction: {end_time - start_time} seconds")

""" #Create the knowledge graph:
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
    query = "What was the relation between adolf hitler and einstein?"
    top_k_triplets = retriever.retrive_triplets_from_knowledgegraph(query, hops=3, top_k=4)
    #Filter the top_k_triplets:
    top_k_triplets = retriever.filter_relevant_triplets(query, top_k_triplets, filter_portion=0.3)
    

    #Print the retrieved triplets for the query:
    if verbose: print(f"Retrieved triplets for the query: {query}")
    if verbose:
        for triplet in top_k_triplets:
            print(triplet)
    end_time = time.time()
    if verbose: print(f"Time taken for graph retrieval: {end_time - start_time} seconds")

#Use the answer model to respond to the query using the retrieved information:
    print('\n\n')
    print(f'The answering the question:\n{query}')
    answer_model.generate(query=query, triplets=top_k_triplets) """
    
