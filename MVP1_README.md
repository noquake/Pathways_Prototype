# MVP 1: Public Clinical Chat + Role-Aware Access

## Overview

This MVP implements a foundational platform for clinical pathway chat with:
- Public chat interface (no login required)
- Authentication via Keycloak
- Role-based access control (public, practitioner, hr, admin)
- Practitioner conversation memory
- HR and Admin dashboards (read-only metadata)

## Architecture

### Services

1. **Frontend** (React) - Port 3000
2. **Backend API** (FastAPI) - Port 8000
3. **Keycloak** (Auth) - Port 8080
4. **PostgreSQL** (Database) - Port 5432
5. **Traefik** (Reverse Proxy) - Port 80/443

### Directory Structure

```
Pathways/
├── docker-compose.mvp1.yml    # Main compose file for MVP 1
├── services/
│   ├── backend/                # FastAPI backend service
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── database_schema.sql
│   └── frontend/               # React frontend
│       ├── src/
│       │   ├── App.js
│       │   ├── components/
│       │   └── pages/
│       ├── package.json
│       └── Dockerfile
├── keycloak/
│   └── config/                 # Keycloak realm configuration
└── MVP1_README.md
```

## Setup Instructions

### 1. Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ (for local backend development)

### 2. Environment Variables

Create a `.env` file in the project root:

```bash
JWT_SECRET=your-secret-key-here
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
ACME_EMAIL=admin@example.com
```

### 3. Initialize Database Schema

```bash
# Connect to database and run schema
docker exec -i pathways_db psql -U admin -d pathways < services/backend/database_schema.sql
```

### 4. Start Services

```bash
# Start all services
docker-compose -f docker-compose.mvp1.yml up -d

# View logs
docker-compose -f docker-compose.mvp1.yml logs -f

# Stop services
docker-compose -f docker-compose.mvp1.yml down
```

### 5. Access Services

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Keycloak Admin**: http://localhost:8080/auth/admin (admin/admin)
- **Traefik Dashboard**: http://localhost:8081

## Default Users

Keycloak is pre-configured with test users:

- **practitioner1** / password123 (Practitioner role)
- **hr1** / password123 (HR role)
- **admin1** / password123 (Admin role)

## API Endpoints

### Public Endpoints

- `GET /health` - Health check
- `POST /chat/public` - Public chat (no auth required)

### Protected Endpoints

- `POST /chat/practitioner` - Practitioner chat (requires JWT)
- `GET /history/{user_id}` - Get chat history (requires JWT)

## Frontend Routes

- `/` - Public chat
- `/login` - Login page
- `/practitioner` - Practitioner chat (requires auth)
- `/hr` - HR dashboard (requires hr/admin role)
- `/admin` - Admin dashboard (requires admin role)

## Security Notes

- JWT tokens are validated on backend
- MFA can be enabled in Keycloak (not enforced in MVP 1)
- No PHI stored in chat logs
- HTTPS should be configured in production (Traefik with Let's Encrypt)

## Next Steps

1. Integrate existing RAG pipeline (`query.py`) with backend
2. Implement actual LLM calls (OpenAI/Gemini/Ollama)
3. Add MFA enforcement for staff roles
4. Implement HR/Admin API endpoints for metadata
5. Add proper error handling and logging
6. Set up production TLS certificates

## Troubleshooting

### Keycloak not starting
- Check database connection
- Verify Keycloak logs: `docker logs pathways_keycloak`

### Backend connection errors
- Verify database is healthy: `docker ps`
- Check backend logs: `docker logs pathways_backend`

### Frontend not connecting
- Check CORS settings in backend
- Verify API_URL environment variable

