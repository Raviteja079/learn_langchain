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

Prompt Template = prepares the input
Model            = generates the response

**Next: LCEL**
**Runnable**
A Runnable is basically a LangChain component that knows how to:
receive an input → do some work → produce an output

User question
      ↓
 ┌────┴─────┐
 ↓          ↓
Retriever   Query analysis
 ↓          ↓
docs        intent
 └────┬─────┘
      ↓
    combine

RunnableParallel = send the same input through multiple processing paths and collect their outputs.

PromptTemplate → make prompts reusable and dynamic
Runnable/LCEL → reusable/composable processing pipeline or make the whole LLM workflow reusable and composable,

**structured output**
Structured output is not mainly about making the LLM's answer "better."
It's about making the LLM's output predictable and machine-readable.

Tools → structured input to functions
Structured output → structured data from the LLM

Both are ways of making LLMs work reliably with normal software.

**RAG**
Your documents
      ↓
   Split into chunks
      ↓
   Create embeddings
      ↓
   Store in vector DB
      ↓
User asks question
      ↓
Convert question → embedding
      ↓
Search similar chunks
      ↓
Relevant chunks
      ↓
LLM + question + chunks
      ↓
Answer

Retrieve relevant knowledge first, then ask the LLM to answer using that knowledge.

Retrieval → find relevant information
Augmented → add it to the LLM's context
Generation → LLM generates the answer

That's why chunking improves retrieval precision.