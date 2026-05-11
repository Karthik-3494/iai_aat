import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

txt_files = [
    "data/holiday_list.txt",
    "data/exam_schedule.txt",
    "data/canteen_menu.txt"
]

documents = []

for file in txt_files:
    loader = TextLoader(file, encoding='utf-8')
    docs = loader.load()
    documents.extend(docs)

print(f"Loaded {len(documents)} pages")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector store ready")


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)


prompt = PromptTemplate(
    template="""
You are a helpful college campus assistant.

Answer ONLY from the provided context.

If the answer is not found in the context,
say:
"I could not find that information in the college documents."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS so your React frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data structure for incoming questions
class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_bot(question: Question):
    # 1. Retrieve relevant chunks from the vector store
    retrieved_docs = retriever.invoke(question.text)

    # 2. Combine the retrieved text into one context string
    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # 3. Format the prompt with the context and the user's question
    final_prompt = prompt.invoke({
        "context": context_text,
        "question": question.text
    })

    # 4. Get the answer from the LLM
    answer = llm.invoke(final_prompt)

    # 5. Return the answer as JSON to the frontend
    return {"answer": answer.content}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
