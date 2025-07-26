from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os
import argparse
import sys
from dotenv import load_dotenv

# Load your OpenAI key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


parser = argparse.ArgumentParser(description="Ask questions about a PDF file.")
parser.add_argument(
    "pdf",
    nargs="?",
    help="Path to the PDF file (positional argument)",
)
parser.add_argument(
    "--pdf",
    dest="pdf_flag",
    help="Path to the PDF file (optional flag --pdf)"
)

args = parser.parse_args()

# Use whichever argument was provided
pdf_path = args.pdf_flag or args.pdf

if not pdf_path:
    parser.error("You must specify a PDF file either positionally or with --pdf.")

# Check if file exists
if not os.path.isfile(pdf_path):
    print(f"❌ Error: File '{pdf_path}' does not exist.")
    sys.exit(1)


# Step 3: Load the PDF
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