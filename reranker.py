from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma 

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_texts(
    documents,
    embedding = embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)


docs = retriever.invoke(
    "Tell me about the company's leave policy."
)

question = "Tell me about the company's leave policy."

pairs = [
    [question, doc.page_content]
    for doc in docs
]

scores = reranker.predict(pairs)

for doc, score in zip(docs, scores):
    print("\nScore:", score)
    print(doc.page_content)