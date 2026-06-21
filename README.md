# Aviation Safety Incident Retrieval System (RAG)

**Tools:** Python · sentence-transformers · HuggingFace Transformers · PyTorch · scikit-learn · pandas · NumPy  
**Domain:** Aviation Safety · NLP · Generative AI · Information Retrieval  
**Dataset:** NASA ASRS (Aviation Safety Reporting System) — 2,634 real incident reports  
**Context:** Individual Assignment — Generative AI Applications (MSc Business Analytics), University of Warwick  
**Status:**  Complete

---

##  Project Overview

Aviation is one of the safest forms of transport — but that safety record is the result of decades of proactive risk management and continuous learning from past incidents. The NASA ASRS database contains thousands of anonymised, narrative-rich incident reports submitted by pilots, air traffic controllers, and flight crew.

The problem: these reports are **unstructured, variable, and hard to search systematically**. Traditional keyword search misses nuance — "runway excursion" and "left the paved surface" mean the same thing but wouldn't match.

**Solution:** I built a full **Retrieval-Augmented Generation (RAG) pipeline** that combines:
- **Dense semantic retrieval** to find relevant incident reports
- **Cross-encoder reranking** to improve result precision
- **FLAN-T5 generation** to produce structured, grounded summaries

---

##  System Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Embedding    │  ← BAAI/bge-large-en-v1.5
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Vector Search      │  ← Cosine similarity (scikit-learn)
│  Top 20 candidates  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Reranking          │  ← BAAI/bge-reranker-base (cross-encoder)
│  Top 5 results      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Answer Generation  │  ← FLAN-T5-large
└─────────────────────┘
    │
    ▼
  Grounded Safety Summary
```

---

##  Key Results

### Query: "Altitude deviations during climb"
Retrieved 5 highly relevant incidents covering:
- Miscommunication between pilots and ATC
- Pilot distraction and high cockpit workload
- Incorrect autopilot settings
- Errors in entering minimum vectoring altitudes

### Query: "What kinds of ATC miscommunication result in safety events?"
Retrieved 5 incidents across Taxi, Descent, and Cruise phases covering:
- Missed or blocked radio calls increasing runway incursion risk
- Ambiguous controller language causing crew confusion
- Static interference distorting communications
- Task overload delaying critical messages

**Key strength:** The system retrieved semantically relevant incidents even when query wording differed from original report language — demonstrating the advantage of dense retrieval over keyword matching.

---

##  Repository Structure

```
aviation-safety-rag/
│
├── data/
│   └── asrs_cleaned.csv              # Cleaned NASA ASRS incident reports
│
├── notebooks/
│   └── aviation_rag_pipeline.py      # Full RAG pipeline
│
└── README.md
```

---

##  Methodology

### 1. Data Preparation
- Loaded 2,634 real NASA ASRS aviation incident reports
- Dropped records with missing Narrative field
- Fields used: `Narrative`, `FlightPhase`, `EventType`

### 2. Semantic Chunking
- Each narrative split at sentence boundaries, capped at 500 characters
- Preserves complete semantic units — important for aviation reports where incidents have multiple components (initial condition → human error → equipment fault → resolution)

### 3. Embedding
- Used `BAAI/bge-large-en-v1.5` to encode each chunk into a dense vector
- Chosen for strong retrieval performance on domain-specific text
- Stored in memory as NumPy arrays (scalable to FAISS/ChromaDB for production)

### 4. Retrieval
- Query embedded using same model
- Cosine similarity computed against all chunk embeddings
- Top 20 candidates retrieved

### 5. Reranking
- `BAAI/bge-reranker-base` cross-encoder scores each query-chunk pair in full context
- Significantly reduces noise for broad or ambiguous queries
- Returns top 5 most relevant chunks

### 6. Generation
- `FLAN-T5-large` generates a structured summary from retrieved context
- Instruction-tuned for zero-shot and few-shot tasks
- Deterministic output for consistency

---

## Testing Summary

| Query Type | Example | Result |
|---|---|---|
|  Specific | "Altitude deviations during climb" | Highly relevant, phase-specific results |
|  Broad | "ATC miscommunication safety events" | Semantically rich, multi-phase retrieval |
|  Too vague | "runway" | Mixed results across landing/taxi/incursion |
|  Long narrative | "Unstable approaches" | Truncation caused context loss in generation |

---

##  Limitations & Proposed Improvements

| Issue | Proposed Fix |
|---|---|
| Short/vague queries return mixed results | Flight phase metadata filtering |
| Long reports exceed model input limits | Improved chunking strategy |
| Occasional semantic drift in generation | Prompt engineering improvements |
| Domain gap in embedding model | Fine-tune on aviation-specific QA pairs |
| No quantitative evaluation | Add ROUGE/BLEU scoring with expert-reviewed QA pairs |

**Future ideas:**
- Hybrid retrieval (dense + BM25 keyword)
- Multi-modal RAG (radar plots, voice transcripts, checklists)
- Real-time deployment with FAISS + GPU acceleration
- Interactive feedback loop for chunk relevance ratings

---

##  Why This Matters

This project demonstrates skills directly relevant to **AI/ML analyst, data scientist, and consulting analyst roles**:
- End-to-end NLP pipeline design
- Practical application of LLMs to real-world unstructured data
- Critical evaluation of AI system strengths and failure modes
- Domain-specific problem framing (aviation safety)
- Clear communication of technical findings to non-technical audiences

---

##  Tech Stack

```python
# Core libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Models used
embedding_model = "BAAI/bge-large-en-v1.5"
reranker_model  = "BAAI/bge-reranker-base"
generator_model = "google/flan-t5-large"
```

---

##  How to Run

```bash
# Install dependencies
pip install pandas sentence-transformers transformers torch scikit-learn numpy

# Run the pipeline
python notebooks/aviation_rag_pipeline.py

# Enter your query when prompted
#   Enter your aviation safety query: altitude deviations during climb
```

---

## 🔗 Data Source

NASA Aviation Safety Reporting System (ASRS):
https://asrs.arc.nasa.gov/search/database.html

---

##  Contact

**Bishal Ranjan Bora**  
[LinkedIn](https://linkedin.com/in/bishalbora) | [Email](mailto:bora.vishal.15@gmail.com) | [GitHub](https://github.com/bishalbora1998)
