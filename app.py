from dotenv import find_dotenv, load_dotenv
print(find_dotenv())

load_dotenv('.env')

import os
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
# llm = ChatOpenAI(model="gpt-4-turbo-preview")
loader = PyMuPDFLoader("fifty.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_data(data):
    return "\n\n".join(dat.page_content for dat in data)


rag_chain = (
    {"context": retriever | format_data, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(force=True)  # Get data from POST request
    question = data['question']  # Get the question from the data
    confirmation = data.get('confirmation')  # Get the confirmation from the data

    # Get the confirmation value from the environment variables
    expected_confirmation = os.getenv('CONFIRMATION')

    # Check if the confirmation variable matches your expected value
    if confirmation == expected_confirmation:
        response = rag_chain.invoke(question)  # Use your function to get the response
        return jsonify(response)  # Return the response as JSON
    else:
        return jsonify({"error": "Invalid confirmation value."}), 403
@app.route('/')
def hello():
    return 'hello world'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)