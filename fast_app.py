from dotenv import load_dotenv
import os
import json
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import chainlit as cl

from ingest import Ingest

# setx OPENAI_API_KEY "your_openai_api_key_here"

# Access the Hugging Face API token from an environment variable
# huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
# if huggingface_token is None:
#    raise ValueError("Hugging Face token is not set in environment variables.")

openai_api_key = "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"
if openai_api_key is None:
    raise ValueError("OAI token is not set in environment variables.")


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
english_embedding_model = "text-embedding-3-large"
czech_embedding_model = "Seznam/simcse-dist-mpnet-paracrawl-cs-en"

czech_store = "stores/czech_512"
english_store = "stores/english_512"

ingestor = Ingest(
    openai_api_key=openai_api_key,
    chunk=512,
    overlap=256,
    czech_store=czech_store,
    english_store=english_store,
    czech_embedding_model=czech_embedding_model,
    english_embedding_model=english_embedding_model,
)


def prompt_en():
    prompt_template_en = """You are electrical engineer and you answer users ###Question.

    #Your answer has to be helpful, relevant and closely related to the user's ###Question.
    #Provide as much literal information and transcription from the #Context as possible. 
    #Only use your own words to connect, clarify or explain the information!
    #If you don't know the answer, just say that you don't know, don't try to make up an answer.

    ###Context: {context}
    ###Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt_en = PromptTemplate(
        template=prompt_template_en, input_variables=["context", "question"]
    )
    print("\n Prompt ready... \n\n")
    return prompt_en


def prompt_cz():
    prompt_template_cz = """Jste elektroinženýr a odpovídáte uživatelům na ###Otázku.

    #Vaše odpověď musí být užitečná, relevantní a úzce souviset s uživatelovou ###Otázkou.
    #Poskytněte co nejvíce doslovných informací a přepisů z #Kontextu.
    #Použijte vlastní slova pouze pro spojení, objasnění nebo vysvětlení informací!
    #Pokud odpověď neznáte, prostě řekněte, že to nevíte, nepokoušejte se vymýšlet odpověď.

    ###Kontext: {context}
    ###Otázka: {question}

    Níže vraťte pouze užitečnou odpověď a nic jiného.
    Užitečná odpověď:
    """
    prompt_cz = PromptTemplate(
        template=prompt_template_cz, input_variables=["context", "question"]
    )
    print("\n Prompt ready... \n\n")
    return prompt_cz


@app.get("/", response_class=HTMLResponse)
def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ingest_data")
async def ingest_data(folderPath: str = Form(...), language: str = Form(...)):
    # Determine the correct data path and store based on the language
    if language == "czech":
        print("\n Czech language selected....\n\n")
        ingestor.data_czech = folderPath
        ingestor.ingest_czech()
        message = "Czech data ingestion complete."
    else:
        print("\n English language selected....\n\n")
        ingestor.data_english = folderPath
        ingestor.ingest_english()
        message = "English data ingestion complete."

    return {"message": message}


@app.post("/get_response")
async def get_response(query: str = Form(...), language: str = Form(...)):
    print(language)
    if language == "czech":
        prompt = prompt_cz()
        print("\n Czech language selected....\n\n")
        embedding_model = czech_embedding_model
        persist_directory = czech_store
        model_name = embedding_model
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        prompt = prompt_en()
        print("\n English language selected....\n\n")
        embedding_model = english_embedding_model  # Default to English
        persist_directory = english_store
        embedding = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embedding_model,
        )

    vectordb = FAISS.load_local(persist_directory, embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    chain_type_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    response = qa_chain(query)

    for i in response["source_documents"]:
        print(f"\n{i}\n\n")

    print(response)

    answer = response["result"]
    source_document = response["source_documents"][0].page_content
    doc = response["source_documents"][0].metadata["source"]
    response_data = jsonable_encoder(
        json.dumps({"answer": answer, "source_document": source_document, "doc": doc})
    )

    res = Response(response_data)
    return res
