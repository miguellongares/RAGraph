import os
import re
import string
import time
from collections import Counter

from datasets import load_dataset
from dotenv import load_dotenv
from smolagents import InferenceClientModel

from src.graph_builder import KnowledgeGraph
from src.retriever import GraphRetriever
from src.answerer import AnswererModel
from src.extractor import ExtractionModel


# ==============================
# Text Normalization
# (official HotpotQA evaluation style)
# ==============================

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


# ==============================
# Exact Match
# ==============================

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ==============================
# F1 Score
# ==============================

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


# ==============================
# Evaluation Loop
# ==============================

def evaluate(model_answer, retriever, dataset, max_samples):
    total_em = 0
    total_f1 = 0

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break

        question = sample["question"]
        gold_answer = sample["answers"]["text"][0]

        # Retrieve graph context
        retrieved_triplets = retriever.retrive_triplets_from_knowledgegraph(
            question,
            hops=3,
            top_k=4
        )
        filtered_triplets = retriever.filter_relevant_triplets(
            question,
            retrieved_triplets,
            threshold=0.4
        )
        # Generate answer
        prediction = model_answer.generate(
            query=question,
            triplets=filtered_triplets
        )
        print(prediction)
        # Metrics
        em = exact_match(prediction, gold_answer)
        f1 = f1_score(prediction, gold_answer)

        total_em += em
        total_f1 += f1

        print(f"\nExample {i+1}")
        print("Q:", question)
        print("Gold:", gold_answer)
        print("Pred:", prediction)
        print("EM:", em, "F1:", f1)

    n = max_samples

    return {
        "EM": total_em / n,
        "F1": total_f1 / n
    }


# ==============================
# Main evaluation script
# ==============================

#The SQuAD 2.0 dataset contains more than 60.000 samples of answerd questions with a correspoing context. Each context has 5 questions, meaning that the samples form 1 to 5 share the same context and so forth. In order to test the perfomance of the model the first 100 questions will be used to measure the performance and the 20 corresponding context will at the begining all be merged to form from the beginning a knowledge graph. 

# --- CONFIGURATION ---
N_SAMPLES = 100
VERBOSE = True
MERGED_TEXT_PATH = "data/all_text.txt"
TRIPLETS_PATH = "outputs/raw_triplets.txt"

def main():

    load_dotenv()

    # Initialize models
    model_extraction = InferenceClientModel(
        model_id='Qwen/Qwen2.5-Coder-7B-Instruct'
    )
    model_answer = InferenceClientModel(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        provider="novita"
    )
    extractor = ExtractionModel(model_extraction)
    answerer = AnswererModel(model_answer)


    #Load dataset
    print("Loading SQuAD 2.0 dataset...")
    dataset = load_dataset("squad", split="train").select(range(N_SAMPLES))
    #Merge all the context in a text file in all_text.txt:
    context = dataset[0:100:5]['context']
    with open(MERGED_TEXT_PATH, 'w', encoding='utf-8') as text_file:
        text_file.write("\n\n".join(context))


    # Triplet Extraction (with caching)
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


    # Graph Construction
    start_time = time.time()
    kg = KnowledgeGraph()
    kg.build_graph(triplets)
    if VERBOSE: 
        print(f"-> Graph built in {time.time() - start_time:.2f}s")

    retriever = GraphRetriever(kg)

    print("Starting evaluation...")

    start = time.time()

    results = evaluate(
        answerer,
        retriever,
        dataset,
        max_samples=5   # change to 500+ later
    )

    print("\n========================")
    print("FINAL RESULTS")
    print("========================")
    print("Exact Match:", results["EM"])
    print("F1 Score:", results["F1"])
    print("Time:", time.time() - start)


if __name__ == "__main__":
    main()