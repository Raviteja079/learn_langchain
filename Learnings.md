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
RunnableParallel → sends the same input into multiple branches and combines their outputs into a dictionary.
reranked_context → takes the question → retrieves + reranks → formats the top docs into context.
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

   Similarity search: "Give me the most relevant documents."

MMR: "Give me relevant documents, but don't waste my slots on near-duplicates."

**fetch_k** : Because MMR needs options to choose from.
fetch_k=10 → first collect 10 candidate chunks

Imagine:

fetch_k = 10**

1. Very relevant, duplicate
2. Very relevant, duplicate
3. Very relevant, duplicate
4. Relevant, different
5. Relevant, different
...
10. Somewhat relevant

MMR can look at those 10 and say:

"Instead of taking three almost-identical chunks, I'll take #1, #4, #5 because they give broader information."

So the mental model is:

fetch_k = candidate pool
k = final number of documents


retriever.invoke()
        ↓
      docs          ← returned value
        ↓
 warning/log       ← automatically displayed by LangChain


 We've now covered:

k → how many final documents to return
score_threshold → minimum relevance requirement
MMR → relevance + diversity
fetch_k → candidate pool for MMR


**Reranking** 
Because reranking is usually more computationally expensive.
100,000 documents
       ↓
expensive reranker
       ↓
😵

Instead:

100,000 documents
       ↓
cheap vector search
       ↓
Top 20 candidates
       ↓
expensive reranker
       ↓
Top 3

Vector search vs reranker

Think of them like this:

Vector search:

"Out of this huge library, which 20 books look like they might be about my question?"

Reranker:

"Now that you've given me these 20 candidates, carefully examine each one against the actual question and tell me which 3 are most relevant."

So:

Retriever = broad + fast

Reranker = narrow + more precise


(question, document) → relevance score
**build a tiny reranking experiment now. with *cross-encoder reranker* ***

One thing I want you to notice: the reranker sees the question and document together, unlike our embedding search where they were independently converted to vectors and compared.
Normal:
Question
  ↓
Retriever
  ↓
Docs
  ↓
LLM


With reranking:
Question
  ↓
Retriever
  ↓
Candidate Docs
  ↓
Reranker  ← EXTRA STEP
  ↓
Docs + relevance scores
  ↓
Sort/filter
  ↓
Best Docs
  ↓
LLM


for reranker scores : For this particular cross-encoder, higher score = more relevant

still model-dependent judgment.

That's exactly why we evaluate retrieval systems using datasets/metrics rather than assuming a reranker is perfect. That's a later production topic. 👍

-2.023 > -5.368, so:

-2.023 → more relevant ✅
-5.368 → less relevant


100,000 docs
     ↓
Chroma similarity search
     ↓
50 candidates          ← cheap
     ↓
Cross-encoder
     ↓
Top 5                  ← accurate


So the key idea is:

Vector search = fast candidate finder
Reranker = slower, more precise judge of those candidates

#Reranker in RAG pipeline
                 Chroma
Question ───────────────────→ Candidate Docs
                                  │
                                  ↓
                         CrossEncoder
                    (Question + each Doc)
                                  │
                                  ↓
                            Relevance scores
                                  │
                                  ↓
                              Sort ↓
                                  │
                                  ↓
                              Top 3 Docs


                              Question
                                 ↓
                              Chroma retriever
                                 ↓
                              rerank_documents()
                                 ↓
                              Top 3 Documents
                                 ↓
                              format_docs()
                                 ↓
                              Prompt
                                 ↓
                              LLM





                              LCEL
                              ↓
                              question + retrieved docs
                              ↓
                              rerank_documents(question, docs)
                                    ↓
                                 create pairs
                                    ↓
                                 CrossEncoder scores
                                    ↓
                                 sort
                                    ↓
                                 top 3 docs


"Tell me about the company's leave policy."
                │
                ▼
       RunnableParallel
          /          \
         /            \
   question         retriever
      │                 │
      │              Chroma
      │                 │
      │              6 docs
      └────────┬────────┘
               ▼
       rerank_documents()
               │
               ├── create pairs
               ├── CrossEncoder scores
               ├── sort
               └── top 3

```retrieval_and_rerank = (
    RunnableParallel(
        question=RunnablePassthrough(),
        docs=retriever
    )
    | RunnableLambda(
        lambda x: rerank_documents(x["question"], x["docs"])
    )
)```

check point: 


Where we are in LangChain

Completed:

Models + Messages ✅
Prompt Templates ✅
Structured Output ✅
LCEL + Runnables ✅
Tools + Tool Calling + Agents ✅
RAG fundamentals ✅
Embeddings
Chroma/vector stores
Retrievers
Chunking
k
Similarity search
Threshold
MMR
Reranking ← currently here

Still remaining:
7. Advanced RAG → hybrid search, query transformation, etc.
8. Memory + persistence
9. LangGraph
10. MCP
11. LangSmith + evaluation
12. Guardrails + production patterns
13. 🚀 Build the resume-worthy project

So I'd say we're roughly 50–55% through the learning curriculum, with the most important practical foundations already covered.



                    Question
                       │
                       ▼
              ┌─────────────────┐
              │ Chroma Retriever│
              └────────┬────────┘
                       │
                 Candidate docs
                       │
                       ▼
              ┌─────────────────┐
              │   CrossEncoder  │
              │    Reranker     │
              └────────┬────────┘
                       │
                    Top 3
                       │
                       ▼
                 format_docs()
                       │
                       ▼
             ┌───────────────────┐
             │ Context + Question│
             └─────────┬─────────┘
                       │
                       ▼
                    Prompt
                       │
                       ▼
                      LLM
                       │
                       ▼
                     Answer



Question
   ↓
Chroma — fast candidate retrieval
   ↓
6 candidate documents
   ↓
CrossEncoder — relevance reranking
   ↓
Top 3 documents
   ↓
format_docs()
   ↓
Prompt
   ↓
Mistral
   ↓
Grounded answer


***broad retrieval first → precise reranking second → generation last. 🔥***



**Hybrid Search**

                    Question
                       ↓
             ┌─────────┴─────────┐
             ↓                   ↓
       Vector Search         BM25 Search
       (meaning)             (keywords)
             ↓                   ↓
         candidates          candidates
             └─────────┬─────────┘
                       ↓
                 Combine results
                       ↓
                  Reranker
                       ↓
                    Top docs
                       ↓
                      LLM

   One thing to absorb today
Think:
Vector search asks:  
"What documents mean something similar?"

BM25 asks:  
"What documents contain/resemble these important words?"

Hybrid asks:  
"Can I use both types of evidence?"

And this is why production RAG systems often don't rely on only vector similarity.

learningn evidence:
***Vector search → MMR/threshold → Reranking → Hybrid search*** 
Each one addresses a different weakness of basic retrieval.
✅ Hybrid-search diagram
✅ Working hybrid retrieval code
✅ 3 test queries + retrieved results
✅ 3-sentence explanation of why hybrid search helps

✅ Covered
1. Models + Messages
2. Prompt Templates
3. Structured Output
4. LCEL + Runnables
5. Tools + Tool Calling + Agents
6. RAG fundamentals
7. Retrieval strategies
   - Similarity
   - Threshold
   - MMR
   - Reranking
   - Hybrid search ← currently learning
🔜 Remaining
8. Advanced RAG
   - Query transformation
   - Multi-query
   - HyDE
   - Parent/child retrieval
   - Contextual compression
9. Memory + persistence
10. LangGraph
11. MCP
12. LangSmith / tracing / evaluation
13. Guardrails + production patterns
14. 🚀 Build the resume-worthy AI full-stack project
So we're roughly 55–60% through the learning curriculum.


BM25's IDF calculation is based on how rare a term is across the whole document collection.
In your collection:
"is"             → appears in only Doc 5
"inv-938271"     → appears in only Doc 4
It only sees tokens + their statistical importance.
*In a real BM25 implementation, you'd normally also use things like stop-word removal (is, the, of, etc.), better tokenization, stemming/normalization, and a much larger corpus.

This document was highly ranked by Chroma, and this one was highly ranked by BM25 — let's combine those signals."

That's the next concept: rank fusion.
A common approach is Reciprocal Rank Fusion (RRF).

Chroma rank + BM25 rank

        ↓
   combined score
        ↓
     final ranking


enumerate() gives us the rank.
scores.get(doc, 0) gives us the doc's existing accumulated score, or 0 if we've never seen it.
The = line adds the new rank-based points to that accumulated score.

Why not use CrossEncoder from the beginning?
Because CrossEncoder is relatively expensive.

10,000 documents
      ↓
Chroma/BM25
      ↓
50 candidates
      ↓
CrossEncoder
      ↓
Top 3