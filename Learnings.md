agent.invoke() starts an orchestration loop.

agent = create_agent(
    model=model,
    tools=[
        get_order_status,
        get_customer_email,
        cancel_order
    ]
) So LangChain constructs the appropriate model request containing things such as:


System/instructions
        +
Available tools and their schemas
        +
Conversation messages
        ↓
      LLM



An agent is basically:

LLM decision
     ↓
Tool execution
     ↓
Result
     ↓
LLM decision
     ↓
Tool execution
     ↓
...


prompt.invoke() doesn't call the LLM.
It only creates the formatted messages.

