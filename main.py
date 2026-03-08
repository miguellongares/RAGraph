import os
import time
from dotenv import load_dotenv
from smolagents import InferenceClientModel

# Local modular imports
from src.extractor import ExtractionModel, merge_text_files
from src.graph_builder import KnowledgeGraph      # Fixed name
from src.retriever import GraphRetriever          # Fixed typo
from src.answerer import AnswererModel

# --- CONFIGURATION ---
DATA_FOLDER = "data/raw_text_data"
MERGED_TEXT_PATH = "data/all_text.txt"
TRIPLETS_PATH = 'outputs/raw_triplets.txt'
VERBOSE = True
QUERY = "Where did albert einstein live?"

def main():
    load_dotenv()
    
    # 1. Initialize Models
    # Using specific models for specific tasks
    model_extraction = InferenceClientModel(model_id='Qwen/Qwen2.5-Coder-7B-Instruct')
    model_answer = InferenceClientModel(model_id='meta-llama/Meta-Llama-3.1-8B-Instruct', provider='novita')

    extractor = ExtractionModel(model_extraction)
    kg = KnowledgeGraph()
    answerer = AnswererModel(model_answer)

    # 2. Prepare Data
    # Merges all .txt files into one master file for processing
    merge_text_files(data_folder=DATA_FOLDER, output_filename=MERGED_TEXT_PATH)
    
    # 3. Triplet Extraction (with caching)
    start_time = time.time()
    
    if os.path.exists(TRIPLETS_PATH):
        if VERBOSE: print(f"-> Loading existing triplets from {TRIPLETS_PATH}")
        with open(TRIPLETS_PATH, 'r', encoding='utf-8') as f:
            raw_data = f.read()
    else:
        if VERBOSE: print("-> No triplets found. Starting LLM extraction (this may take a while)...")
        extractor.write_triplets(input_path=MERGED_TEXT_PATH, output_path=TRIPLETS_PATH)
        with open(TRIPLETS_PATH, 'r', encoding='utf-8') as f:
            raw_data = f.read()

    # Convert text back to Python list of tuples
    triplets = extractor.get_triplets(raw_data)
    
    if VERBOSE: 
        print(f"Done. Extracted {len(triplets)} triplets in {time.time() - start_time:.2f}s")

    # 4. Graph Construction
    start_time = time.time()
    kg.build_graph(triplets)
    if VERBOSE: 
        print(f"-> Graph built in {time.time() - start_time:.2f}s")

    # 5. Retrieval & Filtering
    # We find the relevant 'neighborhood' in the graph and filter by similarity
    retriever = GraphRetriever(kg)
    
    start_time = time.time()
    # Step A: Graph traversal (hops)
    relevant_context = retriever.retrive_triplets_from_knowledgegraph(QUERY, hops=3, top_k=4)
    # Step B: Semantic filtering (threshold)
    filtered_context = retriever.filter_relevant_triplets(QUERY, relevant_context, threshold=0.4)
    
    if VERBOSE:
        print(f"-> Retrieval took {time.time() - start_time:.2f}s")
        print(f"\nRetrieved Facts for: '{QUERY}'")
        for t in filtered_context: print(f"   - {t}")

    # 6. Final Answering
    print('\n' + '='*30)
    print(f"ANSWERING QUESTION: {QUERY}")
    print('='*30 + '\n')
    
    answerer.generate(query=QUERY, triplets=filtered_context)

if __name__ == "__main__":
    main()