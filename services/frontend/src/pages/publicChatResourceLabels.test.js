import {
	getBaseResourceButtonLabel,
	getResourceButtonLabels,
} from "./publicChatResourceLabels";

describe("publicChatResourceLabels", () => {
	const asthmaLabel = "Asthma (Emergency Department and Inpatient)";

	test("maps ED and inpatient algorithms to concise labels", () => {
		expect(
			getBaseResourceButtonLabel(
				{
					id: "ed",
					label: "Asthma Emergency Department Algorithm",
				},
				asthmaLabel,
			),
		).toBe("ED Algorithm");

		expect(
			getBaseResourceButtonLabel(
				{
					id: "inpatient",
					label: "Asthma Inpatient Algorithm",
				},
				asthmaLabel,
			),
		).toBe("Inpatient Algorithm");
	});

	test("maps educational modules and MPIS score to concise labels", () => {
		expect(
			getBaseResourceButtonLabel(
				{
					id: "module",
					label: "Asthma Pathway Educational Module 9.8.23",
				},
				asthmaLabel,
			),
		).toBe("Educational Module");

		expect(
			getBaseResourceButtonLabel(
				{
					id: "mpis",
					label: "Appendix A Mpis Score Branded 11.5.25 Separate Pdf",
				},
				asthmaLabel,
			),
		).toBe("MPIS Score");
	});

	test("keeps single algorithm labels concise", () => {
		expect(
			getBaseResourceButtonLabel(
				{
					id: "bronchiolitis-algorithm",
					label: "Bronchiolitis Algorithm 10.4.23",
				},
				"Bronchiolitis Clinical Pathway",
			),
		).toBe("Algorithm");
	});

	test("resolves duplicate generic labels with useful qualifiers", () => {
		expect(
			getResourceButtonLabels(
				[
					{ id: "transport", label: "Diabetic Ketoacidosis Dka Transport Algorithm 9 8 22" },
					{ id: "picu", label: "Diabetic Ketoacidosis Dka 3 Picu Algorithm" },
				],
				"Diabetic Ketoacidosis",
			),
		).toEqual({
			transport: "Transport Algorithm",
			picu: "PICU Algorithm",
		});
	});
});
