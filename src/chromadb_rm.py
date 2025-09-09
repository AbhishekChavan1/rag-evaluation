"""
Retriever model for chromadb
Adapted from DSPy's ChromadbRM
Changes:
- Support for local embedding model
- Uses Cohere instead of OpenAI
"""

from typing import Optional, List, Union
import dspy
import backoff
import cohere  # Cohere SDK
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ModuleNotFoundError(
        "You need to install Hugging Face transformers library to use a local embedding model with ChromadbRM."
    ) from exc

if chromadb is None:
    raise ImportError(
        "The chromadb library is required to use ChromadbRM. Install it with `pip install chromadb`"
    )

# -----------------------------
# dotdict replacement
# -----------------------------
class dotdict(dict):
    """Dictionary with dot notation access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# -----------------------------
# ChromadbRM class
# -----------------------------
class ChromadbRM(dspy.Retrieve):
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        cohere_api_key: Optional[str] = None,
        local_embed_model: Optional[str] = None,
        k: int = 7,
    ):
        # Cohere client
        self._cohere_client = cohere.Client(api_key=cohere_api_key) if cohere_api_key else None

        # Initialize ChromaDB
        self._init_chromadb(collection_name, persist_directory)

        # Local model
        if local_embed_model is not None:
            self._local_embed_model = AutoModel.from_pretrained(local_embed_model)
            self._local_tokenizer = AutoTokenizer.from_pretrained(local_embed_model)
            self.use_local_model = True
            self.device = torch.device(
                'mps' if torch.backends.mps.is_available() else 'cpu'
            )
        else:
            self.use_local_model = False

        super().__init__(k=k)

    # -----------------------------
    # Initialize ChromaDB collection
    # -----------------------------
    def _init_chromadb(self, collection_name: str, persist_directory: str) -> chromadb.Collection:
        self._chromadb_client = chromadb.Client(
            Settings(persist_directory=persist_directory, is_persistent=True)
        )
        self._chromadb_collection = self._chromadb_client.get_collection(name=collection_name)
        print(f"Collection Count: {self._chromadb_collection.count()}")
        if self._chromadb_client.list_collections() == []:
            raise ValueError(f"Collection {collection_name} does not exist. Please create it using chromadb.")

    # -----------------------------
    # Get embeddings
    # -----------------------------
    @backoff.on_exception(backoff.expo, Exception, max_time=15)
    def _get_embeddings(self, queries: List[str]) -> List[List[float]]:
        if not self.use_local_model:
            if self._cohere_client is None:
                raise ValueError("Cohere API key not provided for embeddings.")
            response = self._cohere_client.embed(model="embed-english-v2.0", texts=queries)
            return response.embeddings

        # Local model embedding
        encoded_input = self._local_tokenizer(
            queries, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self._local_embed_model(**encoded_input.to(self.device))
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy().tolist()

    # -----------------------------
    # Mean pooling for local model
    # -----------------------------
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # -----------------------------
    # Forward retrieval
    # -----------------------------
    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries
        embeddings = self._get_embeddings(queries)

        k = self.k if k is None else k
        results = self._chromadb_collection.query(query_embeddings=embeddings, n_results=k)

        passages = [dotdict({"long_text": x}) for x in results["documents"][0]]
        return passages
