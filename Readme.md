Documentaion for the prof of concept of a Graph RAG symstem:

We will use:

    An agent with an instruct model (≤8 params) for extraction with a proper instruction prompt

    NetworkX for the knowledge graph creation from the extracted triplets. 

    SentenceTransformers small model ('all-MiniLM-L6-v2') for entity similarity and extraction of relevat information from the knowledge graph. 

    Model that answers the user query based on the provided relevant information from the knowledge graph. 

    Pure Python

Improvements that are still to be made: 

    The triplet retrivement is still poor, options to improve:
        -Use a bigger LLM
        -Write a better extraction prompt with better examples
        -Create a multi-step prcess: first get objects then extract all triplets
        -Use an other model/agent to check if the answers are correct
        -Using an agent is better than just using the model?
    
    Many triplets are similar, use a similarity model to merge all the nodes that are related to each other.




Extras: 
    The ussage of the transformer library to download and handle HF models was too uneficient to use and therefore switched to the use of Ollama and smoalagents for the information extraction of texts. 