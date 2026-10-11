from rank_bm25 import BM25Okapi
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

def tokenize(text):
    return re.findall(r'\b[\w-]+\b', text.lower())



documents = [
    "Employees receive 20 days of paid leave every year.",
    "Employees can work from home on Fridays.",
    "The company provides health insurance to all full-time employees.",
    "Order INV-938271? was shipped yesterday.",
    "Order INV-472819 is currently processing."
]

tokenized_documents = [
    tokenize(doc)
    for doc in documents
]
bm25 = BM25Okapi(tokenized_documents)
query = "What is the status of INV-938271?"
tokenized_query = tokenize(query)

bm25_results = bm25.get_top_n(tokenized_query,documents, n=3)

print("=== BM25 ===")

for doc in bm25_results:
    print(doc)


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vector_store = Chroma.from_texts(
    documents,
    embedding=embeddings
)

chroma_results = vector_store.similarity_search(
    query,
    k=3
)
print("\n=== Chroma ===")

for doc in chroma_results:
    print(doc.page_content)

# enumerate() gives us the rank.
# scores.get(doc, 0) gives us the doc's existing accumulated score, or 0 if we've never seen it.
# The = line adds the new rank-based points to that accumulated score.

# bm25_results = [
#     "Order INV-472819 is currently processing.",
#     "Order INV-938271? was shipped yesterday.",
#     "The company provides health insurance to all full-time employees."
# ]
# chroma_results = [Document(id='45ee2123-7019-4416-8346-fa6b13f7f0f3', metadata={}, page_content='Order INV-472819 is currently processing.'), Document(id='7a982125-a1ab-4e53-8778-e54c9d0f8b1a', metadata={}, page_content='Order INV-938271? was shipped yesterday.'), Document(id='73bd5775-23a6-4d9b-899f-42a81fb1e8a7', metadata={}, page_content='Employees can work from home on Fridays.')]

def rrf_fusion(result_lists, k=60):
    scores = {}
    for results in result_lists:
        for rank,doc in enumerate(results, start = 1):
            # !important
            scores[doc] = scores.get(doc,0) + 1/(k + rank)

    ranked_docs = sorted( 
            scores.items(),
            key=lambda x: x[1],
            reverse=True
            )
    return ranked_docs

fused_results = rrf_fusion([
    [doc.page_content for doc in chroma_results],
    bm25_results
])

candidate_docs = [
    doc
    for score, doc in fused_results[:3]
]

# for doc, score in fused_results:
#     print(score, "→", doc)
for doc in candidate_docs:
    print(doc, 'doc')


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
pairs = [
    [query, doc]
    for doc in candidate_docs
]

scores = reranker.predict(pairs)

print(scores, 'scores')

ranked_docs = list(zip(candidate_docs, scores))

ranked_docs.sort(
    key=lambda x: x[1],
    reverse=True
)
for doc, score in ranked_docs:
    print(score, "→", doc)  