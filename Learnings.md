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
The next step is important because you'll see how LangChain combines:

chunks → embedding model → vector store

Vector store → stores vectors/documents and can perform similarity search.
Retriever → provides a standard interface to retrieve relevant documents.

A retriever doesn't necessarily have to use similarity search. It can use other retrieval strategies too:

Retriever
 ├── similarity search
 ├── keyword/BM25 search
 ├── hybrid search
 ├── reranking
 └── other custom retrieval logic

 Good. Now we have almost all the pieces for a basic RAG pipeline.

Documents
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Store
   ↓
Retriever
   ↓
Relevant chunks
   ↓
Prompt + chunks
   ↓
LLM
   ↓
Answer


PromptTemplate → creates reusable prompts
Runnable       → standard processing component
Chain / LCEL   → connects components
Embeddings     → converts text → vectors
Vector Store   → stores/searches vectors
Retriever      → gets relevant documents
LLM            → generates answer

                 RAG APPLICATION

Question
   ↓
Retriever
   ↓
Relevant documents
   ↓
PromptTemplate
   ↓
LLM
   ↓
Answer

rag_chain = retriever | prompt | model 

RunnableParallel = fan out one input → multiple processing paths → collect results into a dictionary.



setup = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
)

User Question
      ↓
RunnableParallel
      ↓
 ┌───────────────┐
 │               │
Retriever    Passthrough
 │               │
 ↓               ↓
Context       Question
 └───────┬───────┘
         ↓
       Prompt
         ↓
        LLM

RunnableLambda is one way to transform the retriever's output before passing it forward.

Mental model: RunnableLambda = adapter that makes your normal Python function usable inside an LCEL pipeline.

**Retrieval quality** 🧠

We'll learn:

Why k matters
Why retrieving more documents isn't always better
What similarity search actually returns
Top-k retrieval
Then we'll move into better strategies like MMR, hybrid search, reranking, etc.


we can give k=3 But not just use k=10 or k=100?
So RAG has an important balancing act:

*Retrieve enough information to answer the question, but not so much irrelevant information that you overwhelm the model.*

chunking strategy → retrieval quality are closely connected.


Your retriever isn't thinking:

"This sentence contains the words paid leave, therefore select it."

Instead, the embedding model is representing semantic meaning, and the vector store is comparing the resulting vectors.

That's why it can retrieve something related even when exact words don't match.

And this leads to our next important question:

How do we know whether a retrieved chunk is actually relevant enough to give to the LLM?

That's where similarity thresholds and better retrieval strategies come in. 🔥

Question
   ↓
Embedding model
   ↓
Question vector
   ↓
Compare with every chunk vector
   ↓
Rank by similarity/distance
   ↓
Top-k chunks
   ↓
LLM


So if your database contains zero relevant documents, the retriever can still return something.

For example:

Question: "What is the company's maternity leave policy?"

Available chunks:
1. Paid leave
2. Work from home
3. Health insurance

k = 1
        ↓
Retriever MUST give you something
        ↓
Maybe → Paid leave

This is why we need another mechanism:

Similarity threshold

Instead of:

"Give me the top 3 no matter what."

we can eventually say:

"Give me the top 3 only if they're sufficiently similar




| Method                           | What it is                                | Typical use                           |
| -------------------------------- | ----------------------------------------- | 
| `similarity_search()`            | Direct vector-store search                | **Quick/manual search**               |
| `similarity_search_with_score()` | Direct search + similarity/distance score | **Inspect/debug/filter results**      |
| `as_retriever()`                 | Creates a **Retriever object/interface**  | **Plug retrieval into chains/agents** |




**The important conceptual difference:**  
`similarity_search*` = **perform a search**.  
`as_retriever()` = **create a retrieval interface that can be used as part of a pipeline.** 👍



If our threshold is 0.70:

0.92 → keep
0.81 → keep
0.32 → discard
0.18 → discard

Now the LLM receives only reasonably relevant context
One important catch ⚠️
The meaning/direction of the score depends on the retrieval system and metric.
With our Chroma setup, similarity_search_with_score() gave us a distance, where lower generally means closer/more similar.
So don't blindly write:
if score > 0.7:
until you know what your particular score represents.


The exact useful threshold depends on your embedding model + data + distance configuration. There isn't a universal 0.5 that works for every RAG system.


Question
   ↓
Embedding
   ↓
Search chunks
   ↓
Similarity scores
   ↓
      score >= 0.5 ?
       /       \
     YES       NO
      ↓         ↓
   return     discard

   k = "How many should I retrieve?"

   threshold = "How relevant must it be?"

   ***Next up: **MMR (Maximum Marginal Relevance)** — why sometimes the top 3 similarity results can be nearly duplicates, and how MMR tries to give you relevant but diverse chunks.***