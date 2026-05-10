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


print("\nCampus Helper Bot Ready!")
print("Type 'exit' to stop.\n")

while True:

    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    # Combine context
    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create final prompt
    final_prompt = prompt.invoke({
        "context": context_text,
        "question": question
    })

    # Generate answer
    answer = llm.invoke(final_prompt)

    print("\nAnswer:")
    print(answer.content)
    print("\n" + "-"*50 + "\n")