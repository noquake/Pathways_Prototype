# Local RAG Pipeline with Docling and PostgreSQL + pgvector

A lightweight **Retrieval-Augmented Generation** system using:

- **Docling** for PDF and Word document ingestion and generation of various files for later LLM ingestion (primarily .md at this stage 12/25)
- **SentenceTransformers** for embedding generation
- **Docker** spinning containers for embedding and (future) usage statistics as well as pgadmin for simple monitoring of database activity via docker-compose
- **PostgreSQL + PGVector** preferred database technology for this project and pjvector for injection of embeddings into containers

This project has been designed to be as clear as possible in its execution and debugging. It streams the generation of chunks, embeddings and their injection the docker database to reduce memory usage.

## TO-DO

- citation information support in chunk generation based on document & document page origin
- PUBLIC ASSISTANT
  - Single-page web app (React / Next.js / simple HTML+JS)
  - Chat UI (prompt → response)
  - Stateless back-end calls (no session persistence)
  - Rate-limiting + abuse protection
  - Pathway citation display (doc name + section)
- AUTHENTICATED PRACTITIONER
  - Secure login
  - Role = Practitioner
  - Persistent conversational context
  - Query history tied to user identity
  - Team based collaboration?
  - EMR/EHR integration?
- ADMINISTRATIVE DASHBOARD
  - Login + admin role
  - Usage metrics:
    - Number of queries
    - Most accessed pathways
    - Peak usage times
    - Public vs staff usage split
  - Health monitoring:
    - System uptime
    - Ingestion status
    - Embedding freshness
  - Exportable reports (CSV)

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- PostgreSQL with pgvector extension

---

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install Python packages
pip install -r requirements.txt
```

### 2. Spin up psql database and administration containers

```bash
docker compose up -d
docker ps # check status of containers
```

## Frontend Structure

```

services/frontend/
├── .env # API URLs and Keycloak Client IDs
├── Dockerfile # Containerization for deployment
├── package.json
├── public/ # Static assets (Not processed by React)
│ ├── favicon.ico
│ ├── manifest.json # Metadata for mobile/web app install
│ └── robots.txt # SEO/Crawler instructions
├── src/ # The "Engine Room"
│ ├── index.jsx # React entry point & Keycloak initialization
│ ├── App.jsx # Top-level Routing & Theme Provider
│ ├── index.css # Your Burgundy/Green CSS Variables & Global styles
│ │
│ ├── assets/ # Brand assets (Processed by Vite/Webpack)
│ │ ├── logo-burgundy.svg
│ │ ├── logo-white.svg
│ │ └── icons/ # Medical-specific SVGs (Stethoscopes, Checklists)
│ │
│ ├── components/ # Reusable building blocks
│ │ ├── ui/ # "Atoms": Buttons, Inputs, Modals
│ │ ├── layout/ # "Templates": Nav, Footer, Sidebar
│ │ ├── clinical/ # Specialized: PathwayCards, CitationBox, ChatBubble
│ │ └── roles/ # Role-specific dashboard components
│ │ ├── AdminDashboard.jsx
│ │ ├── HRDashboard.jsx
│ │ └── PractitionerDashboard.jsx
│ │
│ ├── pages/ # Route-level views (The "Destinations")
│ │ ├── Landing.jsx # The public home page
│ │ ├── Dashboard.jsx # The main "Switcher" for authenticated users
│ │ ├── Chat.jsx # Clinical Chat Interface
│ │ ├── Explorer.jsx # Pathway Explorer
│ │ └── Unauthorized.jsx # 403 error page (User has wrong role)
│ │
│ ├── hooks/ # Custom Logic (The "Brain")
│ │ ├── useAuth.js # Custom wrapper for Keycloak
│ │ ├── useRole.js # Helper: const { isAdmin } = useRole()
│ │ └── useChat.js # Logic for managing AI chat streams
│ │
│ ├── services/ # External Communications
│ │ ├── api.js # Axios/Fetch wrapper for backend calls
│ │ └── auth.js # Keycloak configuration/logic
│ │
│ ├── utils/ # Helper functions
│ │ ├── validators.js # Data validation (Clinical code checks)
│ │ └── formatters.js # Date and text formatting
│ │
│ └── routes/ # Access Control
│ ├── AppRoutes.jsx # Central route definitions
│ └── ProtectedRoute.jsx # Wrapper to check Login/Roles
```
