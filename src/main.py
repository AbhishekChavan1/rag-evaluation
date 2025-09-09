# -----------------------------
# main.py
# -----------------------------
import os
import cohere
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# -----------------------------
# Config
# -----------------------------
os.environ["COHERE_API_KEY"] = "d0i4FtQ5r472xRGiQGJ8nxNGDsBGsh7nWyW9FcRm"  # Replace with your key
EXIT_PROMPT = "exit"

# -----------------------------
# Cohere LLM Wrapper
# -----------------------------
class CohereLM:
    def __init__(self, api_key, model="command", max_tokens=100, temperature=0.5):
        self.client = cohere.Client(api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.generations[0].text

# -----------------------------
# Local Embedding Model
# -----------------------------
class LocalEmbedder:
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model.to(self.device)

    def embed(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# -----------------------------
# ChromaDB Retriever
# -----------------------------
class ChromaRetriever:
    def __init__(self, collection_name, persist_directory, embedder):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory, is_persistent=True))
        self.collection = self.client.get_collection(name=collection_name)
        self.embedder = embedder
        print(f"Collection Count: {self.collection.count()}")

    def retrieve(self, query: str, k=5):
        query_embedding = self.embedder.embed([query])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return results["documents"][0]

# -----------------------------
# RAG System
# -----------------------------
class RAG:
    def __init__(self, llm, retriever, num_passages=5):
        self.llm = llm
        self.retriever = retriever
        self.num_passages = num_passages

    def answer(self, question):
        context_passages = self.retriever.retrieve(question, k=self.num_passages)
        context_text = "\n".join(context_passages)
        prompt = f"Answer the question in 1-5 words.\nContext:\n{context_text}\nQuestion: {question}\nAnswer:"
        return self.llm.generate(prompt)

# -----------------------------
# Main Loop
# -----------------------------
def main():
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    llm = CohereLM(api_key=cohere_api_key)
    embedder = LocalEmbedder()
    retriever = ChromaRetriever(collection_name="test", persist_directory="chroma.db", embedder=embedder)
    rag = RAG(llm, retriever)

    while True:
        print(f"\nEnter the prompt or type {EXIT_PROMPT} to exit:\n")
        prompt = input()
        if prompt.strip().lower() == EXIT_PROMPT:
            print("Exiting...")
            break
        answer = rag.answer(prompt)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()