from smolagents import LiteLLMModel, ChatMessage
import re

path = 'model_instructions/extraction_instruction.txt'

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

