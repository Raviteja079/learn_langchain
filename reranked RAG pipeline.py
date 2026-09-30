from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


documents = documents = [
    "Employees receive 20 days of paid leave every year.",
    "The annual paid leave entitlement for employees is 20 days.",
    "Employees are entitled to 20 days of annual vacation leave.",
    "Employees must get manager approval before taking leave.",
    "Leave requests should be submitted through the HR portal.",
    "Employees can work from home on Fridays.",
    "The company provides health insurance to all full-time employees.",
    "Managers should review leave requests within two working days."
]

embedding = HuggingFaceEmbeddings(
    model = "all-MiniLM-L6-v2"
)
vector_store = Chroma.from_texts(documents, embedding=embedding)

# question = "Tell me about the company's leave policy."
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# docs = retriever.invoke(question)

# pairs = [[question, doc.page_content] for doc in docs]
# # [
# #   ["Tell me about leave policy.",
# #    "Employees receive 20 days of paid leave every year."],

# #   ["Tell me about leave policy.",
# #    "Employees must get manager approval before taking leave."],

# #   ["Tell me about leave policy.",
# #    "Employees can work from home on Fridays."]
# # ]


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# scores = reranker.predict(pairs)

# # (question, doc 1) → -2.10
# # (question, doc 2) → -2.97
# # (question, doc 3) → -5.36
# # Higher score = more relevant for this particular CrossEncoder.

# ranked_docs = list(zip(docs, scores))
# # Document A → -2.10
# # Document B → -2.97
# # Document C → -5.36

# print(ranked_docs)

# ranked_docs.sort(
#     key=lambda x: x[1], reverse=True
# )



# top_docs = [doc 
#             for doc, score in ranked_docs[:3]]
# print(top_docs)









def rerank_documents(question, docs):
    pairs = [
        [question, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked_docs = list(zip(docs, scores))

    ranked_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [
        doc
        for doc, score in ranked_docs[:3]
    ]

    return top_docs

# question = "Tell me about the company's leave policy."

# docs = retriever.invoke(question)

# top_docs = rerank_documents(question, docs)

# for doc in top_docs:
#     print("\n---")
#     print(doc.page_content)



retrieval_and_rerank = (
    RunnableParallel(
        question=RunnablePassthrough(),
        docs=retriever
    )
    | RunnableLambda(
        lambda x: rerank_documents(x["question"], x["docs"])
    )
)

# top_docs = retrieval_and_rerank.invoke(
#     "Tell me about the company's leave policy."
# )

# for doc in top_docs:
#     print("\n---")
#     print(doc.page_content)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


reranked_context = (
    retrieval_and_rerank
    | RunnableLambda(format_docs)
)

rag_input = RunnableParallel(
    context=reranked_context,
    question=RunnablePassthrough()
)
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
Context:
{context}

Question:
{question}
""")

model = ChatOllama(model = "mistral")

rag_chain = rag_input | prompt | model

response = rag_chain.invoke(
    "Tell me about the company's leave policy."
)
print(response.content)