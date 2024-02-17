from dotenv import load_dotenv
import os
import json
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain.llms import CTransformers

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

app = FastAPI()
load_dotenv()  
openai_api_key = os.environ.get("OPENAI_API_KEY")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
# embedding_model = "Seznam/simcse-dist-mpnet-czeng-cs-en"
embedding_model = "Seznam/simcse-dist-mpnet-paracrawl-cs-en"
persist_directory = "stores/seznampara_ul_512"

llm = OpenAI(openai_api_key=openai_api_key)
# llm = "model\dolphin-2.6-mistral-7b.Q4_K_S.gguf"
# llm = "neural-chat-7b-v3-1.Q4_K_M.gguf"


"""
### - Local LLM settings - ###

config = {
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2),
}

llm = CTransformers(
    model=llm, model_type="mistral", lib="avx2", **config  # for CPU use
)

### - Local LLM settings end - ###
"""

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

print("\n Prompt ready... \n\n")


model_name = embedding_model
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

print("\n Retrieval Ready....\n\n")


@app.get("/", response_class=HTMLResponse)
def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_response")
async def get_response(query: str = Form(...)):

    chain_type_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    response = qa_chain(query)
    print(response)
    answer = response["result"]
    source_document = response["source_documents"][0].page_content
    doc = response["source_documents"][0].metadata["source"]
    response_data = jsonable_encoder(
        json.dumps({"answer": answer, "source_document": source_document, "doc": doc})
    )

    res = Response(response_data)
    return res
