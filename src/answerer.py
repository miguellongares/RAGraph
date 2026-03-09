import ollama

path = 'model_instructions/answer_instruction.txt'
path = 'model_instructions/answer_instruction_squad.txt'

with open(path, 'r') as f:
    model_instructions = f.read()

#Define the default model for the answering system:
model='qwen2.5:1.5b-Instruct'

class AnswererModel():
    def __init__(self, model=model):
        self.model = model 

    #Takes the list of relevant tripletes and returns it as context in txt form:
    def turn_tripletes_to_context(self, triplets):
        context = []
        for triplet in triplets:
            context.append(' '.join(triplet + ('\n',)))

        context = ''.join(context)
        return context

    #With the retrived query-relevant-information generate an answer:
    def generate(self, query, triplets):
        context = self.turn_tripletes_to_context(triplets)
        messages=[{'role': 'system', 'content': model_instructions + context},
                        {'role': 'user', 'content': query}]
        
        #Check if the model is local or a HF model:
        if isinstance(self.model, str): #If local olama model  
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
        else: #From HF inference client model
            response = self.model(messages).content
            #print(response)
            
        return response

#run this code to test the model:
if __name__ == "__main__":
    list_of_triplets = [
        ("Albert Einstein", "developed", "Theory of Relativity"),
        ("Albert Einstein", "born in", "Ulm, Germany"),
        ("Albert Einstein", "died in", "Princeton, USA"),
        ("Theory of Relativity", "published in", "1905"),
        ('Hitler', 'born in', 'Braunau am Inn, Austria'),
        ('Hitler', 'perscuted', 'Jews'),
        ('Albert Einstein', 'was', 'Jews'),
    ]
    query = 'What did Albert Einstein have in common with Hitler?'
    answer_model = AnswererModel()
    answer_model.generate(query=query, triplets=list_of_triplets)
            

            


