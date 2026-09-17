from langchain_ollama import ChatOllama
from langchain.tools import tool
from dotenv import load_dotenv 
from langchain.agents import create_agent 
from langchain_core.messages import ToolMessage 




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



model = ChatOllama(model = "mistral")

agent = create_agent(model = model, tools = [get_order_status, cancel_order])

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Check order 101. If it is still processing, cancel it."}
    ]
})

print(result["messages"][-1].content)