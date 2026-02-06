import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "./Chat.css";

function PublicChat({ apiUrl }) {
	const [query, setQuery] = useState("");
	const [messages, setMessages] = useState([]);
	const [loading, setLoading] = useState(false);

	const [showSidebar, setShowSidebar] = useState(true);

	const allCitations = messages
        .flatMap(msg => msg.citations || [])
        // Optional: Filter out duplicates if needed
        .filter((v, i, a) => a.findIndex(t => t.source_file === v.source_file) === i);


	const handleSubmit = async (e) => {
		e.preventDefault();
		if (!query.trim() || loading) return;

		const userMessage = { role: "user", content: query, citations: [] };
		setMessages((prev) => [userMessage, ...prev]);
		setLoading(true);
		setQuery("");

		try {
			const response = await axios.post(`${apiUrl}/chat/public`, {
				query: query,
				// model: 'ollama',
				model: "gemini-pro",
				top_k: 5,
			});

			const assistantMessage = {
				role: "assistant",
				content: response.data.response,
				citations: response.data.citations || [],
				timestamp: response.data.timestamp,
			};
			setMessages((prev) => [assistantMessage, ...prev]);
		} catch (error) {
			console.error("Error:", error);
			const errorMessage = {
				role: "assistant",
				content: "Sorry, I encountered an error. Please try again.",
				citations: [],
			};
			setMessages((prev) => [errorMessage, ...prev]);
		} finally {
			setLoading(false);
		}
	};

	return (
        <div className="chat-page">
            {/* WRAPPER FOR SIDE-BY-SIDE LAYOUT */}
            <div className="chat-layout">
                
                {/* --- LEFT SIDE: CHAT --- */}
                <div className="chat-main">
                    <div className="chat-header" style={{padding: '20px', borderBottom: '1px solid #eee'}}>
                        <h2 style={{margin:0}}>Public Clinical Chat</h2>
                        <p className="disclaimer" style={{margin:'10px 0 0 0'}}>
                            Responses based on clinical pathways. Not medical advice.
                        </p>
                    </div>

                    {/* Input Area (Pinned to Top) */}
                    <form onSubmit={handleSubmit} className="input-container" style={{padding: '20px'}}>
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

                    {/* Messages Area (Scrollable) */}
                    <div className="chat-container">
                        <div className="chat-messages">
                            {messages.map((msg, idx) => (
                                <div key={idx} className={`message ${msg.role}`}>
                                    <div className="message-header">
                                        {msg.role === "user" ? "You" : "Assistant"}
                                    </div>
                                    <div className="message-content">
                                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                                    </div>
                                </div>
                            ))}
                            {loading && <div className="message assistant">Thinking...</div>}
                        </div>
                    </div>
                </div>

                {/* --- RIGHT SIDE: REFERENCES SIDEBAR --- */}
                {showSidebar && (
                    <div className="reference-sidebar">
                        <div className="sidebar-header">
                            Referenced Documents ({allCitations.length})
                        </div>
                        <div className="sidebar-content">
                            {allCitations.length === 0 ? (
                                <p style={{color: '#999', fontStyle: 'italic'}}>
                                    Sources will appear here as you chat.
                                </p>
                            ) : (
                                allCitations.map((cite, idx) => (
                                    <div key={idx} className="reference-card">
                                        <div className="reference-title">📄 {cite.source_file}</div>
                                        {/* Assuming 'cite' has text/snippet content, otherwise remove this div */}
                                        <div className="reference-snippet">
                                            Chunk ID: {cite.chunk_index}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default PublicChat;