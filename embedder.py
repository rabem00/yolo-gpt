import os
import pickle


from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from config import Config

CFG = Config()

class Embedder:
    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    async def storeEmbeds(self, file):
        # file is replaced by CFG.output_file
        loader = CSVLoader(file_path=CFG.output_file, encoding="utf-8",csv_args={"delimiter": ","})  # noqa: E501
        data = loader.load_and_split()
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)
        with open(f"{self.PATH}/{CFG.output_embeddings}", "wb") as f:
            pickle.dump(vectors, f)

    async def getEmbeds(self, file):
        if not os.path.isfile(f"{self.PATH}/{CFG.output_embeddings}"):
            await self.storeEmbeds(file)
        with open(f"{self.PATH}/{CFG.output_embeddings}", "rb") as f:
            vectors = pickle.load(f)
        return vectors
