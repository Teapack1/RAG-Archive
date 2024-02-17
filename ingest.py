from langchain.vectorstores import Chroma
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
    ):
        self.openai_api_key = openai_api_key
        self.chunk = chunk
        self.overlap = overlap
        self.czech_store = czech_store
        self.english_store = english_store
        self.data_czech = data_czech
        self.data_english = data_english

    def ingest_english(self):

        embedding = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-large",
        )

        loader = DirectoryLoader(
            self.data_english,
            show_progress=True,
        )

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk,
            chunk_overlap=self.overlap,
        )
        texts = text_splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=self.english_store,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print("\n English vector Store Created.......\n\n")

    def ingest_czech(self):
        embedding_model = "Seznam/simcse-dist-mpnet-paracrawl-cs-en"
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
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=self.czech_store,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print("\n Czech vector Store Created.......\n\n")


"""       
    
    
    
openai_api_key = "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"
persist_directory = "stores/store_512"
data = "data/"
chunk = 512
overlap = 256

embedding = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-3-large",
    #    model_kwargs={"device": "cpu"},
)

loader = DirectoryLoader(
    data, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader
)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk,
    chunk_overlap=overlap,
)
texts = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"},
)

print("\n Vector Store Created.......\n\n")

"""
