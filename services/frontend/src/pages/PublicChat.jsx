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
	const [selectedResourceId, setSelectedResourceId] = useState("");
	const [pathwayLoadError, setPathwayLoadError] = useState("");
	const [pdfLoading, setPdfLoading] = useState(false);
	const [pdfLoadError, setPdfLoadError] = useState(false);
	const [docScopedQuery, setDocScopedQuery] = useState(false);
	const [useQueryRewriting, setUseQueryRewriting] = useState(false);
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
		if (transcriptRef.current) {
			transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
		}
	}, [messages, loading]);

	const selectedPathway = useMemo(
		() => pathways.find((pathway) => pathway.id === selectedPathwayId),
		[pathways, selectedPathwayId],
	);
	const selectedResource = useMemo(() => {
		if (!selectedPathway) {
			return null;
		}

		return (
			selectedPathway.resources?.find((resource) => resource.id === selectedResourceId) ||
			selectedPathway.resources?.find(
				(resource) => resource.id === selectedPathway.default_resource_id,
			) ||
			selectedPathway.resources?.[0] ||
			null
		);
	}, [selectedPathway, selectedResourceId]);
	const selectedPathwayPdfSrc =
		selectedPathwayId && selectedResource
			? `${apiUrl}/pathways/${selectedPathwayId}/pdf?resource_id=${encodeURIComponent(selectedResource.id)}`
			: "";

	useEffect(() => {
		if (!selectedPathway) {
			setSelectedResourceId("");
			return;
		}

		setSelectedResourceId(selectedPathway.default_resource_id);
		setDocScopedQuery(false);
	}, [selectedPathway]);

	useEffect(() => {
		if (selectedPathway && selectedResource) {
			setPdfLoading(true);
			setPdfLoadError(false);
			return;
		}

		setPdfLoading(false);
		setPdfLoadError(false);
	}, [selectedPathway, selectedResource]);

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
			const history = messages.slice(-6).map((m) => ({
				role: m.role === "user" ? "user" : "assistant",
				content: m.content,
			}));

			const response = await axios.post(`${apiUrl}/chat/public`, {
				query: trimmedQuery,
				model: "gemini",
				top_k: 5,
				conversation_history: history,
				use_query_rewriting: useQueryRewriting,
				...(docScopedQuery && selectedResource?.medembed_id
					? { pathway_id: selectedResource.medembed_id }
					: { pathway_tag: selectedPathwayId || null }),
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

					<div className="chat-toggles">
						{selectedResource?.medembed_id && (
							<button
								type="button"
								className={`chat-toggle-btn ${docScopedQuery ? "active" : ""}`}
								onClick={() => setDocScopedQuery((prev) => !prev)}
							>
								{docScopedQuery
									? `Doc: ${selectedResource.label}`
									: "This doc only"}
							</button>
						)}
						<button
							type="button"
							className={`chat-toggle-btn ${useQueryRewriting ? "active" : ""}`}
							onClick={() => setUseQueryRewriting((prev) => !prev)}
						>
							{useQueryRewriting ? "Context-aware: ON" : "Context-aware: OFF"}
						</button>
					</div>

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
								<div className="chat-icon assistant">A</div>
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
									{msg.role === "user" ? "Q" : "A"}
								</div>
								<div className="chat-content">
									<ReactMarkdown>{msg.content}</ReactMarkdown>
								</div>
							</div>
						))}

						{loading && (
							<div className="chat-card assistant">
								<div className="chat-icon assistant">A</div>
								<div className="chat-content">
									<p>Thinking...</p>
								</div>
							</div>
						)}
					</div>
				</section>

				<aside className="pathway-panel">
					<div className="pathway-panel-header">
						<div className="pathway-panel-title">
							{selectedPathway
								? `${selectedPathway.label} Pathway`
								: "Pathway PDF"}
						</div>
						{selectedPathway?.resources?.length > 1 && (
							<div className="pathway-resource-selector">
								{selectedPathway.resources.map((resource) => (
									<button
										key={resource.id}
										type="button"
										className={`pathway-resource-button ${
											selectedResource?.id === resource.id ? "active" : ""
										}`}
										onClick={() => setSelectedResourceId(resource.id)}
									>
										{resource.label}
									</button>
								))}
							</div>
						)}
						{selectedResource?.pdf_url && (
							<a
								className="pathway-panel-link"
								href={selectedResource.pdf_url}
								target="_blank"
								rel="noreferrer"
							>
								Open full PDF
							</a>
						)}
					</div>
					<div className="pathway-preview-shell">
						{selectedPathway ? (
							<>
								{pdfLoading && !pdfLoadError && (
									<div className="pathway-preview-status">
										<p>Loading pathway PDF...</p>
									</div>
								)}
								{pdfLoadError ? (
									<div className="pathway-preview-empty">
										<p>Pathway PDF unavailable.</p>
									</div>
								) : (
									<iframe
										key={`${selectedPathwayId}:${selectedResource?.id || ""}`}
										src={selectedPathwayPdfSrc}
										title={`${selectedPathway.label} ${selectedResource?.label || "Pathway"} PDF`}
										className="pathway-preview-frame"
										onLoad={() => setPdfLoading(false)}
										onError={() => {
											setPdfLoading(false);
											setPdfLoadError(true);
										}}
									/>
								)}
							</>
						) : (
							<div className="pathway-preview-empty">
								<p>Select a pathway to view the PDF.</p>
							</div>
						)}
					</div>
				</aside>
			</div>
		</div>
	);
}

export default PublicChat;
