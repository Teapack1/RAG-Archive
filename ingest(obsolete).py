from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredFileLoader,
)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)


persist_directory = "stores/test_512"
data = "data\czech"
chunk = 512
overlap = 128
# embedding_model = "Seznam/simcse-dist-mpnet-czeng-cs-en"
embedding_model = "Seznam/simcse-dist-mpnet-paracrawl-cs-en"

model_name = embedding_model
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

"""
loader = CSVLoader(
    file_path="data/emails.csv",
    encoding="utf-8",
    csv_args={
        "delimiter": ";",
    },
)

"""

loader = DirectoryLoader(data, show_progress=True)


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
