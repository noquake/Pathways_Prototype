import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './Chat.css';

function PractitionerChat({ apiUrl, keycloak }) {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    // Load conversation history
    if (keycloak && keycloak.tokenParsed) {
      const userId = keycloak.tokenParsed.sub;
      loadHistory(userId);
    }
  }, [keycloak]);

  const loadHistory = async (userId) => {
    try {
      const response = await axios.get(`${apiUrl}/history/${userId}`, {
        headers: {
          Authorization: `Bearer ${keycloak.token}`
        }
      });
      setHistory(response.data);
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMessage = { role: 'user', content: query, citations: [] };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    const currentQuery = query;
    setQuery('');

    try {
      const response = await axios.post(`${apiUrl}/chat/practitioner`, {
        query: currentQuery,
        model: 'ollama',
        top_k: 5
      }, {
        headers: {
          Authorization: `Bearer ${keycloak.token}`
        }
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response,
        citations: response.data.citations || [],
        timestamp: response.data.timestamp
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        citations: []
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-page">
      <div className="chat-container">
        <h2>Practitioner Chat</h2>
        <p className="disclaimer">
          Your conversation history is saved for context. Previous interactions will inform responses.
        </p>
        
        {history.length > 0 && (
          <div className="history-section">
            <h3>Recent History</h3>
            {history.slice(0, 3).map((item, idx) => (
              <div key={idx} className="history-item">
                <strong>Q:</strong> {item.query}
              </div>
            ))}
          </div>
        )}

        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-header">
                {msg.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div className="message-content">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              {msg.citations && msg.citations.length > 0 && (
                <div className="citations">
                  <strong>Sources:</strong>
                  {msg.citations.map((cite, cIdx) => (
                    <div key={cIdx} className="citation-item">
                      <strong>{cite.source_file}</strong> (Chunk {cite.chunk_index})
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-content">Thinking...</div>
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="input-container">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about clinical pathways..."
            className="query-input"
            disabled={loading}
          />
          <button type="submit" className="submit-button" disabled={loading || !query.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default PractitionerChat;

