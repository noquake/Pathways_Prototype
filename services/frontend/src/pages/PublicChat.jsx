import React, { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "./PublicChat.css";
import { getResourceButtonLabels } from "./publicChatResourceLabels";

const FEEDBACK_COMMENT_MAX_LENGTH = 500;
const FEEDBACK_OPTIONS = [
	"Answer satisfactory",
	"Wrong (risky)",
	"wrong (minor)",
	"missing info (risky)",
	"missing info (minor)",
	"misc.",
];

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
	const [openFeedbackMenuId, setOpenFeedbackMenuId] = useState("");
	const [openFeedbackEditorId, setOpenFeedbackEditorId] = useState("");
	const [feedbackDrafts, setFeedbackDrafts] = useState({});
	const [feedbackSelections, setFeedbackSelections] = useState({});
	const [savingFeedbackId, setSavingFeedbackId] = useState("");
	const [feedbackError, setFeedbackError] = useState({
		queryId: "",
		message: "",
	});
	const transcriptRef = useRef(null);
	const feedbackTextareaRef = useRef(null);

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

	useEffect(() => {
		if (!openFeedbackMenuId && !openFeedbackEditorId) {
			return undefined;
		}

		const handleKeyDown = (event) => {
			if (event.key !== "Escape") {
				return;
			}

			if (openFeedbackEditorId) {
				const openMessage = messages.find((message) => message.queryId === openFeedbackEditorId);
				if (openMessage) {
					setFeedbackDrafts((prev) => ({
						...prev,
						[openMessage.queryId]: openMessage.feedbackComment || "",
					}));
					setFeedbackSelections((prev) => ({
						...prev,
						[openMessage.queryId]: openMessage.userFeedback || "",
					}));
				}
			}
			setOpenFeedbackMenuId("");
			setOpenFeedbackEditorId("");
			setFeedbackError({
				queryId: "",
				message: "",
			});
		};

		window.addEventListener("keydown", handleKeyDown);
		return () => window.removeEventListener("keydown", handleKeyDown);
	}, [messages, openFeedbackMenuId, openFeedbackEditorId]);

	useEffect(() => {
		if (openFeedbackEditorId && feedbackTextareaRef.current) {
			feedbackTextareaRef.current.focus();
		}
	}, [openFeedbackEditorId]);

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
	const resourceButtonLabels = useMemo(
		() =>
			getResourceButtonLabels(selectedPathway?.resources || [], selectedPathway?.label || ""),
		[selectedPathway],
	);
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

	const getFeedbackDraft = (message) =>
		Object.prototype.hasOwnProperty.call(feedbackDrafts, message.queryId)
			? feedbackDrafts[message.queryId]
			: message.feedbackComment || "";

	const getFeedbackSelection = (message) =>
		Object.prototype.hasOwnProperty.call(feedbackSelections, message.queryId)
			? feedbackSelections[message.queryId]
			: message.userFeedback || "";

	const resetFeedbackEditor = (message) => {
		setFeedbackDrafts((prev) => ({
			...prev,
			[message.queryId]: message.feedbackComment || "",
		}));
		setFeedbackSelections((prev) => ({
			...prev,
			[message.queryId]: message.userFeedback || "",
		}));
	};

	const openFeedbackEditor = (message, nextSelection) => {
		if (openFeedbackEditorId && openFeedbackEditorId !== message.queryId) {
			const openMessage = messages.find((item) => item.queryId === openFeedbackEditorId);
			if (openMessage) {
				resetFeedbackEditor(openMessage);
			}
		}

		setOpenFeedbackMenuId("");
		setOpenFeedbackEditorId(message.queryId);
		setFeedbackDrafts((prev) => ({
			...prev,
			[message.queryId]: getFeedbackDraft(message),
		}));
		setFeedbackSelections((prev) => ({
			...prev,
			[message.queryId]:
				typeof nextSelection === "string" ? nextSelection : getFeedbackSelection(message),
		}));
		setFeedbackError({
			queryId: "",
			message: "",
		});
	};

	const closeFeedbackEditor = (message) => {
		setOpenFeedbackEditorId("");
		resetFeedbackEditor(message);
		setFeedbackError({
			queryId: "",
			message: "",
		});
	};

	const handleFeedbackSubmit = async (message) => {
		const currentDraft = getFeedbackDraft(message);
		const trimmedComment = currentDraft.trim();
		const selectedFeedback = getFeedbackSelection(message);
		if ((!trimmedComment && !selectedFeedback) || savingFeedbackId) {
			return;
		}

		setSavingFeedbackId(message.queryId);
		setFeedbackError({
			queryId: "",
			message: "",
		});

		try {
			const payload = {};
			if (selectedFeedback) {
				payload.user_feedback = selectedFeedback;
			}
			if (trimmedComment) {
				payload.feedback_comment = trimmedComment;
			}
			const response = await axios.patch(
				`${apiUrl}/queries/${encodeURIComponent(message.queryId)}/feedback`,
				payload,
			);
			const savedComment = response.data.feedback_comment || message.feedbackComment || "";
			const savedUserFeedback =
				response.data.user_feedback || message.userFeedback || "";

			setMessages((prev) =>
				prev.map((item) =>
					item.queryId === message.queryId
						? {
								...item,
								feedbackComment: savedComment,
								userFeedback: savedUserFeedback,
							}
						: item,
				),
			);
			setFeedbackDrafts((prev) => ({
				...prev,
				[message.queryId]: savedComment,
			}));
			setFeedbackSelections((prev) => ({
				...prev,
				[message.queryId]: savedUserFeedback,
			}));
			setOpenFeedbackEditorId("");
			setOpenFeedbackMenuId("");
		} catch (error) {
			const detail =
				error.response?.data?.detail || "Could not save feedback. Please try again.";
			setFeedbackError({
				queryId: message.queryId,
				message: detail,
			});
		} finally {
			setSavingFeedbackId("");
		}
	};

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
				queryId: response.data.query_id || "",
				feedbackComment: "",
				userFeedback: "",
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
					queryId: "",
					feedbackComment: "",
					userFeedback: "",
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
								<div className="chat-body">
									{msg.role === "assistant" && msg.queryId && (
										<div className="chat-actions">
											<button
												type="button"
												className="chat-action-trigger"
												aria-label="Open feedback actions"
												aria-haspopup="menu"
												aria-expanded={openFeedbackMenuId === msg.queryId}
												onClick={() => {
													if (openFeedbackEditorId) {
														const openMessage = messages.find(
															(item) => item.queryId === openFeedbackEditorId,
														);
														if (openMessage) {
															resetFeedbackEditor(openMessage);
														}
													}
													setOpenFeedbackMenuId((currentId) =>
														currentId === msg.queryId ? "" : msg.queryId,
													);
													setOpenFeedbackEditorId("");
													setFeedbackError({
														queryId: "",
														message: "",
													});
												}}
											>
												...
											</button>
												{openFeedbackMenuId === msg.queryId && (
													<div className="chat-action-menu" role="menu">
															<button
																type="button"
																className="chat-action-menu-button"
																role="menuitem"
																onClick={() => openFeedbackEditor(msg)}
															>
																{msg.feedbackComment || msg.userFeedback
																	? "Edit feedback"
																	: "Leave comment"}
															</button>
														</div>
													)}
												</div>
											)}
											<div className="chat-content">
												<ReactMarkdown>{msg.content}</ReactMarkdown>
											</div>
											{msg.role === "assistant" &&
												(msg.feedbackComment || msg.userFeedback) &&
												openFeedbackEditorId !== msg.queryId && (
												<div className="feedback-note feedback-note-success">
												Feedback saved
											</div>
										)}
											{msg.role === "assistant" &&
												msg.queryId &&
												openFeedbackEditorId === msg.queryId && (
												<div className="feedback-panel">
													<div className="feedback-category-row">
														{FEEDBACK_OPTIONS.map((option) => (
															<button
																key={option}
																type="button"
																className={`feedback-category-button ${
																	getFeedbackSelection(msg) === option ? "selected" : ""
																}`}
																onClick={() => openFeedbackEditor(msg, option)}
																disabled={savingFeedbackId === msg.queryId}
															>
																{option}
															</button>
														))}
													</div>
														<label
															className="feedback-label"
															htmlFor={`feedback-comment-${msg.queryId}`}
														>
															Add an optional comment about this answer
													</label>
													<textarea
														id={`feedback-comment-${msg.queryId}`}
														ref={feedbackTextareaRef}
														className="feedback-textarea"
														value={getFeedbackDraft(msg)}
														onChange={(event) => {
															const nextValue = event.target.value.slice(
																0,
																FEEDBACK_COMMENT_MAX_LENGTH,
															);
															setFeedbackDrafts((prev) => ({
																...prev,
																[msg.queryId]: nextValue,
															}));
															if (feedbackError.queryId === msg.queryId) {
																setFeedbackError({
																	queryId: "",
																	message: "",
																});
															}
														}}
														rows={4}
														maxLength={FEEDBACK_COMMENT_MAX_LENGTH}
														placeholder="What should we improve about this answer?"
													/>
													<div className="feedback-panel-footer">
														<div className="feedback-counter">
															{getFeedbackDraft(msg).length}/
															{FEEDBACK_COMMENT_MAX_LENGTH}
														</div>
														<div className="feedback-panel-actions">
															<button
																type="button"
																className="feedback-button feedback-button-secondary"
																onClick={() => closeFeedbackEditor(msg)}
																disabled={savingFeedbackId === msg.queryId}
															>
																Cancel
															</button>
															<button
																type="button"
																className="feedback-button feedback-button-primary"
																onClick={() => handleFeedbackSubmit(msg)}
																disabled={
																	savingFeedbackId === msg.queryId ||
																	(!getFeedbackDraft(msg).trim() &&
																		!getFeedbackSelection(msg))
																}
															>
																{savingFeedbackId === msg.queryId
																	? "Saving..."
																	: "Save feedback"}
															</button>
														</div>
													</div>
													{feedbackError.queryId === msg.queryId && (
														<div className="feedback-note feedback-note-error">
															{feedbackError.message}
														</div>
													)}
												</div>
											)}
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
										{resourceButtonLabels[resource.id] || resource.label}
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
