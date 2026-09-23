from langchain_ollama import ChatOllama
from langchain.tools import tool
from dotenv import load_dotenv 
from langchain.agents import create_agent 
from langchain_core.messages import ToolMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



load_dotenv()

# And this is precisely why we first manually implemented:
# LLM → tool call → Python tool → ToolMessage → LLM

@tool 
def calculate(expression: str) -> str: 
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception:
        return f"Invalid Expression"


@tool
def get_order_status(order_id: int) -> str:
    """Get the current status of an order."""
    orders = {
        101: "Shipped",
        102: "Delivered",
        103: "Processing"
    }

    return orders.get(order_id, "Order not found")
@tool
def get_customer_email(customer_id: int) -> str:
    """Get the email address of a customer."""
    customers = {
        1: "john@example.com",
        2: "alice@example.com"
    }
    return customers.get(customer_id, "Customer not found")


@tool
def cancel_order(order_id: int) -> str:
    """Cancel an order."""
    return f"Order {order_id} has been cancelled."

def get_content(response):
    return response.content

def simple_explanation(text):
    return f"Simple explanation of: {text}"

def short_definition(text):
    return f"One-line definition of: {text}"

prompt = ChatPromptTemplate.from_messages([("system","You are a helpful assistant"),("human","Explain {topic} in simple terms")])
messages = prompt.invoke({
    "topic": "RAG"
})
# model = ChatOllama(model = "mistral")
# response = model.invoke(messages)
# chain = prompt | model | get_content
# response = chain.invoke({"topic": "RAG"})
# print(response)

parallel = RunnableParallel(
    simple = RunnableLambda(simple_explanation),
    definition = RunnableLambda(short_definition)
)

result = parallel.invoke("RAG")
# print(result)



# agent = create_agent(model = model, tools = [get_order_status, cancel_order])

# result = agent.invoke({
#     "messages": [
#         {"role": "user", "content": "Check order 101. If it is still processing, cancel it."}
#     ]
# })

# print(result["messages"][-1].content)

class OrderResponse(BaseModel):
    order_id: int
    status: str

model = ChatOllama(model='mistral')
structured_model = model.with_structured_output(OrderResponse)
# result = structured_model.invoke("what is the status of order 101?")
# print(result)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

documents = [
    "Employees can work from home on Fridays.",
    "Employees receive 20 days of paid leave every year.",
    "The company provides health insurance to all full-time employees."
]
question = "How many paid leave days do employees get?"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(documents)
question_embedding = embedding_model.encode(question)
# print(document_embeddings)

similarities = cosine_similarity(
    [question_embedding],
    document_embeddings
)

context = documents[1]

prompt = f"""
Answer the question using the context below

Context: {context}
Question: {question}
"""
# response = model.invoke(prompt)

text = """
Employees receive 20 days of paid leave every year.
Employees can work from home on Fridays.
The company provides health insurance to all full-time employees.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)
# for chunk in chunks: 
#     print("----")
#     print(chunk)

# step1: create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# step 2: create the vector store
vector_store = Chroma.from_texts(
    chunks,
    embedding=embeddings
)

"""that does several things behind the scenes
chunks
  ↓
HuggingFaceEmbeddings
  ↓
vectors
  ↓
Chroma
  ↓
stores text + vectors"""

# step 3: Search 
results = vector_store.similarity_search("How many paid leaves do employees get?", k=3)
# print(results[0].page_content)

# Retreiver 
retriever = vector_store.as_retriever(search_kwargs={"k":3})
# docs = retriever.invoke(
#     "How many paid leave days do employees get?"
# )

# for i, doc in enumerate(docs):
#     print(f"\n--- Document {i+1} ---")
#     print(doc.page_content)


results = vector_store.similarity_search_with_score(
    "How many paid leave days do employees get?",
    k=2
)
# for doc, score in results:
#     print("\n---")
#     print("Score:", score)
#     print("Text:", doc.page_content)


retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.5
    }
)

docs = retriever.invoke(
    "What is the company's maternity leave policy?"
)

for doc in docs:
    print(doc.page_content)



# LLM response 

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
""")

format_docs_runnable = RunnableLambda(format_docs)

setup = RunnableParallel(
    context=retriever | format_docs_runnable,
    question=RunnablePassthrough()
)

rag_chain = setup | prompt | model 

res = rag_chain.invoke("How many leave does a employee can have?")
# print(res.content)

"""
Question
   ↓
RunnableParallel
   ├── retriever → relevant documents
   └── passthrough → original question
             ↓
        {context, question}
             ↓
           Prompt
             ↓
            LLM
"""

