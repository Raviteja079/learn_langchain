from langchain_ollama import ChatOllama
from langchain.tools import tool
from dotenv import load_dotenv 
from langchain.agents import create_agent 
from langchain_core.messages import ToolMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



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
print(result)



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




documents = [
    "Employees can work from home on Fridays.",
    "Employees receive 20 days of paid leave every year.",
    "The company provides health insurance to all full-time employees."
]
question = "How many paid leave days do employees get?"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(documents)
question_embedding = embedding_model.encode(question)
print(document_embeddings)

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
response = model.invoke(prompt)

print(response.content)