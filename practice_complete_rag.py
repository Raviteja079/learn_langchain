from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

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

def rerank_documents(question, docs):
    pairs = [[question, doc.page_content] for doc in docs]
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

#Chunking


#Embedding
embedding = HuggingFaceEmbeddings(
    model = "all-MiniLM-L6-v2"
)

#Chroma db storing - vector store
vector_store = Chroma.from_texts(documents = documents, embedding=embedding)

#Retrieving from DB 
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":6}
)
#retriving from CrossEncoding (filtered records) or reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#Top K 

#format(docs)

#Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
Context:
{context}

Question:
{question}
""")
retrieval_and_rerank = (RunnableParallel(
    question = RunnablePassthrough(),
    docs = retriever
) | RunnableLambda(lambda x: rerank_documents(x["question"], x["docs"])))


def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


reranked_context = (retrieval_and_rerank | RunnableLambda(format_docs))
#LLM
#RunnableParallel = send the same input through multiple processing paths and collect their outputs.
#reranked_context → takes the question → retrieves + reranks → formats the top docs into context.
rag_input = RunnableParallel(
    context = reranked_context,
    question = RunnablePassthrough()
)
model = ChatOllama(model = "mistral")
rag_chain = rag_input | prompt | model
response = rag_chain.invoke(
    "Tell me about the company's leave policy."
)
print(response.content)