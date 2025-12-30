-- MVP 1 Database Schema
-- Extends existing items table with chat logs and user context

-- Chat logs for public users (anonymized)
CREATE TABLE IF NOT EXISTS chat_logs_public (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_hash VARCHAR(64)  -- Hashed IP for basic analytics (no PHI)
);

-- Chat logs for practitioners (with user context)
CREATE TABLE IF NOT EXISTS chat_logs_practitioner (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    response TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_chat_logs_practitioner_user_id ON chat_logs_practitioner(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_logs_practitioner_timestamp ON chat_logs_practitioner(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_chat_logs_public_timestamp ON chat_logs_public(timestamp DESC);

-- Users table (if not managed by Keycloak)
-- Note: In production, users are typically managed by Keycloak
-- This table is for reference/analytics only
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('public', 'practitioner', 'hr', 'admin')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    mfa_enabled BOOLEAN DEFAULT FALSE
);

-- Metadata views for HR/Admin dashboards
CREATE OR REPLACE VIEW practitioner_metadata AS
SELECT 
    u.user_id,
    u.email,
    u.role,
    u.created_at,
    u.last_login,
    COUNT(cp.id) as total_queries,
    MAX(cp.timestamp) as last_query_time
FROM users u
LEFT JOIN chat_logs_practitioner cp ON u.user_id = cp.user_id
WHERE u.role IN ('practitioner', 'hr', 'admin')
GROUP BY u.user_id, u.email, u.role, u.created_at, u.last_login;

-- Aggregate statistics view
CREATE OR REPLACE VIEW aggregate_stats AS
SELECT 
    (SELECT COUNT(*) FROM chat_logs_public) as public_queries,
    (SELECT COUNT(*) FROM chat_logs_practitioner) as practitioner_queries,
    (SELECT COUNT(DISTINCT user_id) FROM chat_logs_practitioner) as active_practitioners,
    (SELECT COUNT(*) FROM items) as total_chunks,
    (SELECT COUNT(DISTINCT source_file) FROM items) as total_pathways;

