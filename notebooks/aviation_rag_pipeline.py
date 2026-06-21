#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 18:22:36 2025

@author: bishalbora
"""

import pandas as pd

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the cleaned dataset

df = pd.read_csv("asrs_cleaned.csv")
df = df.dropna(subset=['Narrative'])
print(f"Loaded {len(df)} real aviation incidents.")

# Step 2: Chunking (optional - but helps split long narratives)

def semantic_chunks(text, max_len=500):
    sentences = text.split('. ')
    chunks, current = [], ''
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += s + '. '
        else:
            chunks.append(current.strip())
            current = s + '. '
    if current:
        chunks.append(current.strip())
    return chunks

# Step 3: Prepare docs
docs = []
for idx, row in df.iterrows():
    chunks = semantic_chunks(row['Narrative'])
    for i, chunk in enumerate(chunks):
        docs.append({
            "text": chunk,
            "flight_phase": row['FlightPhase'],
            "event_type": row['EventType'],
            "id": f"{idx}_{i}"
        })

# Step 4: Embed narratives
print("🔄 Embedding incidents...")

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
doc_embeddings = [model.encode(d['text']) for d in docs]

# Step 5: Reranking model

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

def rerank(query, candidates):
    pairs = [(query, d['text']) for d in candidates]
    inputs = tokenizer([q for q, d in pairs], [d for q, d in pairs],
                       return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        scores = reranker(**inputs).logits.view(-1)
    return [candidates[i] for i in torch.topk(scores, k=5).indices.tolist()]

# Step 6: Get your query
query = input("✈️  Enter your aviation safety query: ")
query_vec = model.encode(query)

# Step 7: Vector search
sims = cosine_similarity([query_vec], doc_embeddings)[0]
top_indices = np.argsort(sims)[-20:][::-1]
retrieved = [docs[i] for i in top_indices]

# Step 8: Rerank and display results
top_docs = rerank(query, retrieved)

print("\n📋 Top 5 Matching Incidents:\n")
for i, doc in enumerate(top_docs, 1):
    print(f"[{i}] (Flight Phase: {doc['flight_phase']}, Type: {doc['event_type']})")
    print(doc['text'])
    print("-" * 80)
