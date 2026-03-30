import React, { act } from "react";
import { createRoot } from "react-dom/client";
import axios from "axios";

jest.mock("axios");
jest.mock("react-markdown", () => ({ children }) => children);

import PractitionerChat from "./PractitionerChat";

describe("PractitionerChat citations", () => {
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

	test("renders only the markdown response and not a separate sources block", async () => {
		axios.get.mockResolvedValue({ data: [] });
		axios.post.mockResolvedValue({
			data: {
				response: "Follow the pathway [1].\n\nSources:\n[1] pathway.pdf",
				citations: [
					{
						chunk_id: "1",
						source_file: "pathway",
					},
				],
				timestamp: "2026-03-30T00:00:00",
			},
		});

		await act(async () => {
			root.render(
				<PractitionerChat
					apiUrl="http://localhost:8000"
					keycloak={{ token: "token", tokenParsed: { sub: "user-1" } }}
				/>,
			);
		});
		await act(async () => {
			await Promise.resolve();
		});

		const input = container.querySelector(".query-input");
		const valueSetter = Object.getOwnPropertyDescriptor(
			window.HTMLInputElement.prototype,
			"value",
		).set;
		await act(async () => {
			valueSetter.call(input, "What should I do?");
			input.dispatchEvent(new Event("input", { bubbles: true }));
		});

		await act(async () => {
			container.querySelector(".submit-button").click();
		});
		await act(async () => {
			await Promise.resolve();
			await Promise.resolve();
		});

		expect(container.textContent).toContain("Sources:");
		expect(container.textContent).toContain("[1] pathway.pdf");
		expect(container.querySelector(".citations")).toBeNull();
	});
});
