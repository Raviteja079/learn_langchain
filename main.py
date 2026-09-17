# from langchain_openai import ChatOpenAI 
from langchain_ollama import ChatOllama
from langchain.tools import tool
from dotenv import load_dotenv 
from langchain.agents import create_agent 
from langchain_core.messages import ToolMessage 




load_dotenv()



# question = input("Ask something: ")
# response = model.invoke(question)
# print(response.content)

@tool 
def calculate(expression: str) -> str: 
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception:
        return f"Invalid Expression"

# print(calculate.invoke("2 + 2"))

@tool
def get_order_status(order_id: int) -> str:
    """Get the current status of an order."""
    orders = {
        101: "Shipped",
        102: "Delivered",
        103: "Processing"
    }

    return orders.get(order_id, "Order not found")

# print(get_order_status.invoke({"order_id": 101}))
model = ChatOllama(model = "mistral")
# model_with_tools = model.bind_tools([get_order_status])
# response = model_with_tools.invoke("what is the status of order 101?")
# print(response.tool_calls)
# tool_call = response.tool_calls[0]
# # print("Tool call:", tool_call)
# tool_result = get_order_status.invoke(tool_call["args"])

# print("Tool result:", tool_result)

# agent = create_agent(model = model, tools=[get_order_status])
# result = agent.invoke({
#     "message": [
#         {
#             "role": "user",
#             "content": "what is the status of order 101?"
#         }
#     ]
# })
# print(result["messages"][-1].content)

# tool_message = ToolMessage(
#     content = tool_result,
#     tool_call_id = tool_call["id"]
# )

# final_response = model_with_tools.invoke([{
#     "role": "user",
#     "content": "what is the status of order 101"
# },response, tool_message])

# print(final_response.content,'final response')





# -------------------------------------------------------let langchain automate loop

agent = create_agent(model = model, tools = [get_order_status])

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the status of order 101?"}
    ]
})

print(result["messages"][-1].content)