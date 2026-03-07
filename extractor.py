from smolagents import LiteLLMModel, ChatMessage
import re
import os
import glob

path = 'model_instructions/new_extraction_instruction.txt'

#get the instructions from the file extraction_instruction.txt inside the folder model_instructions
with open(path, "r") as f:
    instructions = f.read()

# Initialize the local Ollama model
model = LiteLLMModel(
    model_id="ollama_chat/qwen2:7b", 
    api_base="http://localhost:11434", # Default Ollama port
    num_ctx=8192 # Context window
)

class ExtractionModel():
    def __init__(self, model = model):
        self.model = model
    
    def generate(self, prompt):
        messages = [
            ChatMessage(role="system", content=[{"type": "text", "text": instructions}]),
            ChatMessage(role="user", content=[{"type": "text", "text": prompt}])
        ]
        response = self.model.generate(messages=messages)
        return response.content
    
    def get_triplets(self, text):
        #Normalize the text by removing uppercase letters and extra spaces
        text = text.lower().strip()
        # This function can be implemented to further process the model's output and extract triplets in a structured format
        pattern = r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
        return re.findall(pattern, text)
    
    
    #Chunk text without cuting sentences: 
    def chunk_text(self, text, max_tokens = 300):
        # Split text into sentences (handles '.', '!', '?')
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
        # Calculate word count of the sentence
            sentence_words = sentence.split()
            sentence_len = len(sentence_words)
            
            # If adding this sentence exceeds max_tokens, save the current chunk
            if current_length + sentence_len > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.extend(sentence_words)
            current_length += sentence_len

        # Add the final remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    

def merge_text_files(data_folder, output_filename):
    if not os.path.exists(data_folder):
        print(f"Error: Folder '{data_folder}' not found.")
        return
    search_pattern = os.path.join(data_folder, "*.txt")
    files_to_merge = glob.glob(search_pattern)

    count = 0
    with open(output_filename, 'w') as outfile:
        for file_path in files_to_merge:
            try:
                with open.file(file_path, 'r') as f:
                    outfile.write(f.read())
                    outfile.write('/n')# Add separation
            except Exception as err:    
                print(f'Could not read file {file_path}: {err}')
    print('Merge files done')

    

# Run this code to test the model:
if __name__ == "__main__":
    with open("Data/Albert_Einstein.txt", "r") as f:
        text = f.read()
    extraction_model = ExtractionModel(model)
    extracted_triplets = extraction_model.generate(text)
    print(extracted_triplets)
    #Second step: extract the triplets from the model output
    print("Extracting triplets from the model output...")
    triplets = extraction_model.get_triplets(extracted_triplets)
    print(triplets) 

