# Pathways Prototype

A **Retrieval-Augmented Generation (RAG)** system built for the Connecticut Children's use case. The core focus is clean data processing and high-fidelity retrieval — clinical pathways are dense, structured documents and preserving their integrity from ingestion through retrieval is the priority.

---

## Description

This prototype ingests clinical pathway documents, chunks and embeds them, and stores them in a vector database for semantic retrieval. A FastAPI backend handles query routing and LLM interaction, served locally via Uvicorn.

The project has pivoted away from a Docker-based development environment. Running services locally through **Uvicorn** provides significantly clearer debugging and faster iteration. Storage and database concerns have been offloaded to **Supabase**, removing the overhead of managing local containers and freeing the team to focus on what matters most: retrieval quality and prompt engineering.

### Technologies

| Layer                          | Tool                                                                       |
| ------------------------------ | -------------------------------------------------------------------------- |
| Document ingestion             | [Docling](https://github.com/DS4SD/docling) initially -> Claude Sonnet 4.6 |
| Embeddings                     | SentenceTransformers (Various models)                                      |
| Vector storage                 | PostgreSQL + pgvector (via Supabase)                                       |
| Backend                        | FastAPI + Uvicorn                                                          |
| Frontend                       | React                                                                      |
| Code Hygiene + Static analysis | [Vulture](https://github.com/jendrikseipp/vulture)                         |
| Auth (planned)                 | Keycloak                                                                   |

---

## Installation

**Prerequisites:** Python 3.11+, Node.js

```bash
# Clone the repo
git clone https://github.com/noquake/Pathways_Prototype.git
cd Pathways_Prototype

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Set up your environment variables:

```bash
cp .env.example .env
# Fill in your Supabase URL, anon key, and OpenAI/LLM API key
```

---

## Running

```bash
# Start the backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# In a separate terminal, start the frontend
cd services/frontend
npm install
npm run dev
```

The API will be available at `http://localhost:8000` and the frontend at `http://localhost:5173`.

To identify unused code before making changes:

```bash
pip install vulture
vulture services/ --min-confidence 80
```

---

## Next Steps

- **Prompt engineering** — refine system prompts to improve grounding and citation accuracy against clinical pathway content
- **Retrieval tuning** — improve chunking strategy and similarity thresholds to maximize fidelity of returned context
- **Session separation** — isolate conversation context per user session for cleaner multi-turn interactions
- **User accounts** — persistent conversation history and per-user query tracking (most accessed pathways, usage patterns)
- **Role-based access** — public assistant vs. authenticated practitioner vs. admin dashboard, via Keycloak

## Useful Commands

- List of available Pathways

```bash
curl -s http://localhost:8000/pathways | jq
```

- Using a different

```bash
curl -s -X POST http://localhost:8000/chat/public \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Signs of anaphylaxis?",
    "pathway_id": "anaphylaxis",
    "embedding_model": "medcpt",
    "model_name": "gemini-2.0-flash"
  }' | jq
```

- Healthcheck

```bash
curl -s http://localhost:8000/health
```
