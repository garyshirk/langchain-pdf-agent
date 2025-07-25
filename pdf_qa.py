from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv

# Load your OpenAI key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the PDF (replace this with your own file)
pdf_path = "example.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split and embed
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(pages, embeddings)

# Set up the Q&A chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
    retriever=vectorstore.as_retriever()
)

# Ask a question
while True:
    query = input("\nAsk a question about the PDF (or type 'exit' to quit): ")
    if query.lower() in ['exit', 'quit']:
        break
    response = qa_chain.invoke({"query": query})
    print("A:", response)