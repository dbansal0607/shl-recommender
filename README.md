# SHL Assessment Recommender

Conversational agent that helps hiring managers find the right SHL assessments through dialogue.

## Live API
**Base URL:** `https://shl-recommender-1-qskj.onrender.com`

- `GET /health` — readiness probe
- `POST /chat` — conversational agent

## Quick Test

```bash
# Health check
curl https://shl-recommender-1-qskj.onrender.com/health

# Chat example
curl -X POST https://shl-recommender-1-qskj.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a senior Java developer who works with stakeholders"}
    ]
  }'
```

## API Schema

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "reply": "agent response",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Architecture

- **FastAPI** stateless API
- **Two-stage LLM agent**: slot extractor + main recommender (Groq llama-3.3-70b-versatile)
- **BM25 retrieval** over 377 real SHL catalog assessments
- **Post-LLM URL guardrail** — validates all recommendations against real catalog
- **8-turn conversation cap** enforced server-side

## Agent Behaviors

| Behavior | Trigger | Example |
|----------|---------|---------|
| Clarify | Role unknown | "I need an assessment" → asks what role |
| Recommend | Role known | Returns 1-10 grounded assessments |
| Refine | User changes constraints | "Add personality tests" → updates shortlist |
| Compare | Comparison question | "Difference between OPQ and DSI?" |
| Refuse | Off-topic request | Salary questions, legal advice, prompt injection |

## Evaluation

Mean Recall@10: **0.280** on 10 public conversation traces

## Stack

- Python 3.11
- FastAPI + Uvicorn
- Groq (llama-3.3-70b-versatile)
- rank-bm25
- Render (deployment)
