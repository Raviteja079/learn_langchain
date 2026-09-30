from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

documents = [
    "Employees receive 20 days of paid leave every year.",
    "The annual paid leave entitlement for employees is 20 days.",
    "Employees are entitled to 20 days of annual vacation leave.",
    "Employees must get manager approval before taking leave.",
    "Leave requests should be submitted through the HR portal.",
    "Employees can work from home on Fridays.",
    "The company provides health insurance to all full-time employees.",
    "Managers should review leave requests within two working days."
]

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma.from_texts(
    documents,
    embedding = embeddings
)

# 1. Normal searching 
similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)




docs = similarity_retriever.invoke(
    "Tell me about the company's leave policy."
)

print("\n--- Similarity ---")
for doc in docs:
    print(doc.page_content)




# 2. MMR 
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 6
    }
)




docs = mmr_retriever.invoke(
    "Tell me about the company's leave policy."
)

print("\n--- MMR ---")
for doc in docs:
    print(doc.page_content)