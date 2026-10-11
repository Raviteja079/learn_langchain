from rank_bm25 import BM25Okapi
import re

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
    

print(tokenized_query)
scores = bm25.get_scores(tokenized_query)
print(scores)
for doc, score in zip(documents, scores):
    print(score, "→", doc)