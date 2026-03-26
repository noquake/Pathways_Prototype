const VERSION_TOKEN_RE = /\b(?:\d{1,2}[.-]){1,2}\d{2,4}\b|\b\d{8}\b|\b20\d{2}\b/gi;
const NOISE_TOKEN_RE = /\b(?:pdf|branded|final|updated|udpated|separate)\b/gi;
const WHITESPACE_RE = /\s+/g;
const GENERIC_BASE_LABELS = new Set(["Algorithm", "Pathway", "Module", "Educational Module"]);

const QUALIFIER_RULES = [
	{ pattern: /\b(?:emergency department|emergency room|ed|er)\b/i, label: "ED" },
	{ pattern: /\binpatient\b/i, label: "Inpatient" },
	{ pattern: /\btransport\b/i, label: "Transport" },
	{ pattern: /\bpicu\b/i, label: "PICU" },
	{ pattern: /\bprovider\b/i, label: "Provider" },
	{ pattern: /\benglish\b/i, label: "English" },
	{ pattern: /\bspanish\b/i, label: "Spanish" },
];

function escapeRegExp(value) {
	return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeWhitespace(value) {
	return value.replace(WHITESPACE_RE, " ").trim();
}

function getPathwayBaseLabel(pathwayLabel = "") {
	const withoutParenthetical = pathwayLabel.split("(")[0] || pathwayLabel;
	return normalizeWhitespace(
		withoutParenthetical
			.replace(/\bclinical pathway\b/gi, " ")
			.replace(/\bpathway\b/gi, " "),
	);
}

function stripPathwayBaseLabel(value, pathwayLabel) {
	const pathwayBaseLabel = getPathwayBaseLabel(pathwayLabel);
	if (!pathwayBaseLabel) {
		return value;
	}

	return value.replace(new RegExp(escapeRegExp(pathwayBaseLabel), "gi"), " ");
}

function cleanOriginalLabel(label, pathwayLabel) {
	return normalizeWhitespace(
		stripPathwayBaseLabel(label, pathwayLabel)
			.replace(VERSION_TOKEN_RE, " ")
			.replace(NOISE_TOKEN_RE, " ")
			.replace(/\bclinical pathway\b/gi, " ")
			.replace(/\bpathway\b/gi, " ")
			.replace(/\s+[-_/]\s+/g, " ")
			.replace(/\s+[./-]\s*$/g, " ")
			.replace(/\s*\([^)]*\)\s*/g, " "),
	);
}

function getSourceText(resource) {
	return [resource?.label, resource?.doc_name, resource?.id]
		.filter(Boolean)
		.join(" ");
}

function getAppendixLabel(sourceText) {
	const appendixMatch = sourceText.match(/\bappendix\s+([a-z])\b/i);
	if (!appendixMatch) {
		return "";
	}
	return `Appendix ${appendixMatch[1].toUpperCase()}`;
}

function getQualifiers(sourceText) {
	return QUALIFIER_RULES.filter((rule) => rule.pattern.test(sourceText)).map((rule) => rule.label);
}

function getPrimaryQualifier(sourceText) {
	return getQualifiers(sourceText)[0] || "";
}

export function getBaseResourceButtonLabel(resource, pathwayLabel = "") {
	const sourceText = getSourceText(resource);
	const cleanedLabel = cleanOriginalLabel(resource?.label || resource?.doc_name || resource?.id || "", pathwayLabel);
	if (/\bmpis\b/i.test(sourceText)) {
		return "MPIS Score";
	}

	const appendixLabel = getAppendixLabel(sourceText);
	if (appendixLabel) {
		return appendixLabel;
	}

	const qualifier = getPrimaryQualifier(sourceText);
	if (/\beducational module\b|\beducation module\b/i.test(sourceText)) {
		return qualifier ? `${qualifier} Educational Module` : "Educational Module";
	}

	if (/\bmodule\b/i.test(sourceText)) {
		return qualifier ? `${qualifier} Module` : "Module";
	}

	if (/\balgorithm\b/i.test(sourceText)) {
		return qualifier ? `${qualifier} Algorithm` : "Algorithm";
	}

	if (/\bclinical pathway\b|\bpathway\b/i.test(sourceText)) {
		return qualifier ? `${qualifier} Pathway` : "Pathway";
	}

	return cleanedLabel || resource?.label || resource?.doc_name || resource?.id || "";
}

function buildCollisionLabel(resource, pathwayLabel, baseLabel) {
	const sourceText = getSourceText(resource);
	const cleanedLabel = cleanOriginalLabel(resource?.label || resource?.doc_name || resource?.id || "", pathwayLabel);
	if (!GENERIC_BASE_LABELS.has(baseLabel)) {
		return cleanedLabel || baseLabel;
	}

	for (const qualifier of getQualifiers(sourceText)) {
		const candidate = `${qualifier} ${baseLabel}`;
		if (candidate !== baseLabel) {
			return candidate;
		}
	}

	return cleanedLabel || baseLabel;
}

export function getResourceButtonLabels(resources = [], pathwayLabel = "") {
	const baseEntries = resources.map((resource) => ({
		resource,
		label: getBaseResourceButtonLabel(resource, pathwayLabel),
	}));

	const baseCounts = baseEntries.reduce((counts, entry) => {
		counts[entry.label] = (counts[entry.label] || 0) + 1;
		return counts;
	}, {});

	const collisionAdjusted = baseEntries.map((entry) => ({
		resource: entry.resource,
		label:
			baseCounts[entry.label] > 1
				? buildCollisionLabel(entry.resource, pathwayLabel, entry.label)
				: entry.label,
	}));

	const finalCounts = collisionAdjusted.reduce((counts, entry) => {
		counts[entry.label] = (counts[entry.label] || 0) + 1;
		return counts;
	}, {});

	return collisionAdjusted.reduce((labels, entry) => {
		let finalLabel = entry.label;
		if (finalCounts[finalLabel] > 1) {
			const fallbackLabel =
				cleanOriginalLabel(
					entry.resource?.label || entry.resource?.doc_name || entry.resource?.id || "",
					pathwayLabel,
				) || finalLabel;
			finalLabel = fallbackLabel === finalLabel ? `${fallbackLabel} ${entry.resource.id}` : fallbackLabel;
		}
		labels[entry.resource.id] = finalLabel;
		return labels;
	}, {});
}
