import React, { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "./PublicChat.css";

function PublicChat({ apiUrl }) {
	const [query, setQuery] = useState("");
	const [messages, setMessages] = useState([]);
	const [loading, setLoading] = useState(false);
	const [pathways, setPathways] = useState([]);
	const [selectedPathwayId, setSelectedPathwayId] = useState("");
	const [pathwayLoadError, setPathwayLoadError] = useState("");
	const [previewLoadError, setPreviewLoadError] = useState(false);
	const transcriptRef = useRef(null);

	useEffect(() => {
		const loadPathways = async () => {
			try {
				const response = await axios.get(`${apiUrl}/pathways`);
				const pathwayOptions = response.data || [];
				setPathways(pathwayOptions);
				if (pathwayOptions.length > 0) {
					setSelectedPathwayId(pathwayOptions[0].id);
				}
			} catch (error) {
				console.error("Error loading pathways:", error);
				setPathwayLoadError("Could not load pathway list.");
			}
		};
		loadPathways();
	}, [apiUrl]);

	useEffect(() => {
		setPreviewLoadError(false);
	}, [selectedPathwayId]);

	useEffect(() => {
		if (transcriptRef.current) {
			transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
		}
	}, [messages, loading]);

	const selectedPathway = useMemo(
		() => pathways.find((pathway) => pathway.id === selectedPathwayId),
		[pathways, selectedPathwayId],
	);

	const handleSubmit = async (e) => {
		e.preventDefault();
		const trimmedQuery = query.trim();
		if (!trimmedQuery || loading) return;

		const userMessage = {
			role: "user",
			content: trimmedQuery,
			citations: [],
		};
		setMessages((prev) => [...prev, userMessage]);
		setLoading(true);
		setQuery("");

		try {
			const response = await axios.post(`${apiUrl}/chat/public`, {
				query: trimmedQuery,
				model: "gemini",
				top_k: 5,
				pathway_id: selectedPathwayId || null,
			});

			const assistantMessage = {
				role: "assistant",
				content: response.data.response,
				citations: response.data.citations || [],
				timestamp: response.data.timestamp,
			};
			setMessages((prev) => [...prev, assistantMessage]);
		} catch (error) {
			console.error("Error:", error);
			setMessages((prev) => [
				...prev,
				{
					role: "assistant",
					content: "Sorry, I encountered an error. Please try again.",
					citations: [],
				},
			]);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="public-chat-page">
			<div className="public-chat-layout">
				<section className="public-chat-main">
					<h1 className="public-chat-title">Pathways</h1>

					<label className="pathway-select-label" htmlFor="pathway-select">
						Selected Pathway
					</label>
					<select
						id="pathway-select"
						className="pathway-select"
						value={selectedPathwayId}
						onChange={(e) => setSelectedPathwayId(e.target.value)}
						disabled={pathways.length === 0}
					>
						{pathways.length === 0 ? (
							<option value="">
								{pathwayLoadError || "Loading pathways..."}
							</option>
						) : (
							pathways.map((pathway) => (
								<option key={pathway.id} value={pathway.id}>
									{pathway.label}
								</option>
							))
						)}
					</select>

					<form onSubmit={handleSubmit} className="question-form">
						<input
							type="text"
							value={query}
							onChange={(e) => setQuery(e.target.value)}
							placeholder="Hello, How can I help you today?"
							className="question-input"
							disabled={loading}
						/>
						<button
							type="submit"
							className="question-submit"
							disabled={loading || !query.trim()}
							aria-label="Send"
						>
							{loading ? "..." : ">"}
						</button>
					</form>

					<div className="chat-transcript" ref={transcriptRef}>
						{messages.length === 0 && (
							<div className="chat-card assistant">
								<div className="chat-icon assistant">★</div>
								<div className="chat-content">
									<p>
										Select a pathway and ask a question. Responses are limited to
										the selected pathway.
									</p>
								</div>
							</div>
						)}

						{messages.map((msg, idx) => (
							<div key={idx} className={`chat-card ${msg.role}`}>
								<div className={`chat-icon ${msg.role}`}>
									{msg.role === "user" ? "◎" : "★"}
								</div>
								<div className="chat-content">
									<ReactMarkdown>{msg.content}</ReactMarkdown>
								</div>
							</div>
						))}

						{loading && (
							<div className="chat-card assistant">
								<div className="chat-icon assistant">★</div>
								<div className="chat-content">
									<p>Thinking...</p>
								</div>
							</div>
						)}
					</div>
				</section>

				<aside className="pathway-panel">
					<div className="pathway-panel-title">
						{selectedPathway
							? `${selectedPathway.label} Pathway`
							: "Pathway Preview"}
					</div>
					<div className="pathway-preview-shell">
						{selectedPathway && !previewLoadError ? (
							<img
								src={selectedPathway.preview_image_path}
								alt={`${selectedPathway.label} pathway`}
								className="pathway-preview-image"
								onError={() => setPreviewLoadError(true)}
							/>
						) : (
							<div className="pathway-preview-empty">
								<p>Pathway preview not available yet.</p>
							</div>
						)}
					</div>
				</aside>
			</div>
		</div>
	);
}

export default PublicChat;
