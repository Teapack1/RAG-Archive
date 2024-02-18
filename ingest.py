from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)


class Ingest:
    def __init__(
        self,
        openai_api_key=None,
        chunk=512,
        overlap=256,
        czech_store="stores/czech_512",
        english_store="stores/english_512",
        data_czech="data/czech",
        data_english="data/english",
        english_embedding_model="text-embedding-3-large",
        czech_embedding_model="Seznam/simcse-dist-mpnet-paracrawl-cs-en",
    ):
        self.openai_api_key = openai_api_key
        self.chunk = chunk
        self.overlap = overlap
        self.czech_store = czech_store
        self.english_store = english_store
        self.data_czech = data_czech
        self.data_english = data_english
        self.english_embedding_model = english_embedding_model
        self.czech_embedding_model = czech_embedding_model

    def ingest_english(self):

        embedding = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model=self.english_embedding_model,
        )

        loader = DirectoryLoader(
            self.data_english,
            show_progress=True,
            loader_cls=PyPDFLoader,
        )

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk,
            chunk_overlap=self.overlap,
        )
        texts = text_splitter.split_documents(documents)

        vectordb = FAISS.from_documents(
            documents=texts,
            embedding=embedding,
        )
        vectordb.save_local(self.english_store)

        print("\n English vector Store Created.......\n\n")

    def ingest_czech(self):
        embedding_model = self.czech_embedding_model
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        loader = DirectoryLoader(
            self.data_czech,
            show_progress=True,
        )

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk,
            chunk_overlap=self.overlap,
        )

        texts = text_splitter.split_documents(documents)
        vectordb = FAISS.from_documents(
            documents=texts,
            embedding=embedding,
        )
        vectordb.save_local(self.czech_store)

        print("\n Czech vector Store Created.......\n\n")
