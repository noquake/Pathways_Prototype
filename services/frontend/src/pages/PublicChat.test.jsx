import React, { act } from "react";
import { createRoot } from "react-dom/client";
import axios from "axios";

jest.mock("axios");
jest.mock("react-markdown", () => ({ children }) => children);

import PublicChat from "./PublicChat";

describe("PublicChat resource button labels", () => {
	let container;
	let root;

	beforeEach(() => {
		globalThis.IS_REACT_ACT_ENVIRONMENT = true;
		container = document.createElement("div");
		document.body.appendChild(container);
		root = createRoot(container);
	});

	afterEach(async () => {
		await act(async () => {
			root.unmount();
		});
		container.remove();
		jest.clearAllMocks();
	});

	test("renders concise resource labels while keeping the pathway title", async () => {
		axios.get.mockResolvedValue({
			data: [
				{
					id: "asthma",
					label: "Asthma (Emergency Department and Inpatient)",
					default_resource_id: "asthma-ed",
					resources: [
						{
							id: "asthma-ed",
							label: "Asthma Emergency Department Algorithm",
							doc_name: "asthma_emergency_department_algorithm_-_9.8.23",
							pdf_url: "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_emergency_department_algorithm_-_9.8.23.pdf",
						},
						{
							id: "asthma-inpatient",
							label: "Asthma Inpatient Algorithm",
							doc_name: "asthma_inpatient_algorithm_-_08.22.23",
							pdf_url: "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_inpatient_algorithm_-_08.22.23.pdf",
						},
						{
							id: "asthma-mpis",
							label: "Appendix A Mpis Score Branded 11.5.25 Separate Pdf",
							doc_name: "appendix-a-mpis-score-branded-11.5.25-separate-pdf",
							pdf_url: "https://www.connecticutchildrens.org/sites/default/files/2025-11/appendix-a-mpis-score-branded-11.5.25-separate-pdf.pdf",
						},
						{
							id: "asthma-module",
							label: "Asthma Pathway Educational Module 9.8.23",
							doc_name: "asthma_pathway_educational_module_-9.8.23",
							pdf_url: "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_pathway_educational_module_-9.8.23.pdf",
						},
					],
				},
			],
		});

		await act(async () => {
			root.render(<PublicChat apiUrl="http://localhost:8000" />);
		});
		await act(async () => {
			await Promise.resolve();
		});

		const resourceButtons = Array.from(
			container.querySelectorAll(".pathway-resource-button"),
		).map((button) => button.textContent);

		expect(container.querySelector(".pathway-panel-title")?.textContent).toBe(
			"Asthma (Emergency Department and Inpatient) Pathway",
		);
		expect(resourceButtons).toEqual([
			"ED Algorithm",
			"Inpatient Algorithm",
			"MPIS Score",
			"Educational Module",
		]);
		expect(container.textContent).not.toContain("Asthma Emergency Department Algorithm");
	});

	test("renders the assistant response legend without the session sources panel", async () => {
		axios.get.mockResolvedValue({
			data: [
				{
					id: "asthma",
					label: "Asthma",
					default_resource_id: "emergency-department-algorithm",
					resources: [
						{
							id: "emergency-department-algorithm",
							label: "ED Algorithm",
							doc_name: "asthma_emergency_department_algorithm_-_9.8.23",
							pdf_url: "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_emergency_department_algorithm_-_9.8.23.pdf",
							medembed_id: "asthma-emergency-department-algorithm",
						},
						{
							id: "educational-module",
							label: "Educational Module",
							doc_name: "asthma_pathway_educational_module_-9.8.23",
							pdf_url: "https://www.connecticutchildrens.org/sites/default/files/2023-09/asthma_pathway_educational_module_-9.8.23.pdf",
						},
					],
				},
			],
		});
		axios.post.mockResolvedValue({
			data: {
				response:
					"Use the pathway algorithm [1].\n\nSources:\n[1] asthma_emergency_department_algorithm_-_9.8.23.pdf",
				citations: [
					{
						chunk_id: "1",
						chunk_text: "Hidden appendix chunk",
						chunk_length: 20,
						source_file: "asthma_emergency_department_algorithm_-_9.8.23",
						source_docs: ["asthma_emergency_department_algorithm_-_9.8.23"],
						pdf_name: "asthma_emergency_department_algorithm_-_9.8.23.pdf",
						pathway_id: "asthma",
						resource_id: "emergency-department-algorithm",
					},
				],
				timestamp: "2026-03-30T00:00:00",
				query_id: "query-1",
			},
		});

		await act(async () => {
			root.render(<PublicChat apiUrl="http://localhost:8000" />);
		});
		await act(async () => {
			await Promise.resolve();
		});

		const input = container.querySelector(".question-input");
		const valueSetter = Object.getOwnPropertyDescriptor(
			window.HTMLInputElement.prototype,
			"value",
		).set;
		await act(async () => {
			valueSetter.call(input, "What should I use?");
			input.dispatchEvent(new Event("input", { bubbles: true }));
		});
		await act(async () => {
			await Promise.resolve();
		});

		await act(async () => {
			container.querySelector(".question-submit").click();
		});
		await act(async () => {
			await Promise.resolve();
			await Promise.resolve();
		});

		expect(axios.post).toHaveBeenCalled();
		expect(container.textContent).toContain("Sources:");
		expect(container.textContent).toContain(
			"[1] asthma_emergency_department_algorithm_-_9.8.23.pdf",
		);
		expect(container.querySelector(".session-sources")).toBeNull();
	});
});
