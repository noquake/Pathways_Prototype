from __future__ import annotations

import argparse
import os
import pprint
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import Client, create_client

from pathways_catalog_supplement import (
    ACTIVE_PATHWAY_IDS,
    HIDDEN_SOURCE_ALIASES,
    ID_OVERRIDES,
    LEGACY_ORDER,
    RESOURCE_PDF_URL_OVERRIDES,
)

START_URL = "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways"
CONNECTICUT_CHILDRENS_HOST = "www.connecticutchildrens.org"
SECTION_PATH_PREFIX = "/medical-professionals/clinical-pathways/"
PATHWAY_DOCUMENTS_TABLE = "pathway_documents"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "pathways_catalog.py"
PRIMARY_RESOURCE_HINTS = ("algorithm", "clinical-pathway", "clinical pathway")
SECONDARY_RESOURCE_HINTS = (
    "educational-module",
    "educational module",
    "education-module",
    "education module",
    "module",
    "appendix",
    "appendices",
)
ACRONYM_MAP = {
    "aki": "AKI",
    "arch": "ARCH",
    "brue": "BRUE",
    "cap": "CAP",
    "cbc": "CBC",
    "ctach": "CTACH",
    "ed": "ED",
    "hgb": "Hgb",
    "iv": "IV",
    "mpis": "MPIS",
    "ng": "NG",
    "nhlbi": "NHLBI",
    "po": "PO",
}


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(query="", fragment="").geturl()


def normalize_key(value: Optional[str]) -> str:
    if not value:
        return ""

    normalized = unquote(str(value)).strip().lower()
    normalized = normalized.rsplit("/", 1)[-1]
    if normalized.endswith(".pdf"):
        normalized = normalized[:-4]
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized


def pdf_basename(url: str) -> str:
    path = urlparse(url).path
    filename = unquote(Path(path).name)
    if filename.lower().endswith(".pdf"):
        filename = filename[:-4]
    return filename


def titleize_filename(value: str) -> str:
    tokens = [token for token in re.split(r"[-_]+", value) if token]
    if not tokens:
        return value

    words = []
    for token in tokens:
        lower = token.lower()
        if lower in ACRONYM_MAP:
            words.append(ACRONYM_MAP[lower])
        elif token.isdigit():
            words.append(token)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def extract_section_links(html: str, source_url: str = START_URL) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    source_host = urlparse(source_url).netloc
    sections = set()

    for anchor in soup.find_all("a", href=True):
        full_url = canonicalize_url(urljoin(source_url, anchor["href"].strip()))
        parsed = urlparse(full_url)
        if parsed.netloc != source_host:
            continue
        if not parsed.path.startswith(SECTION_PATH_PREFIX):
            continue
        if full_url == canonicalize_url(source_url):
            continue
        sections.add(full_url)

    return sorted(sections)


def extract_page_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    if heading:
        return heading.get_text(" ", strip=True)

    if soup.title:
        return soup.title.get_text(" ", strip=True)

    return ""


def extract_pdf_urls(html: str, section_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    pdf_urls = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        full_url = canonicalize_url(urljoin(section_url, href))
        parsed = urlparse(full_url)
        if parsed.netloc != CONNECTICUT_CHILDRENS_HOST:
            continue
        if not parsed.path.lower().endswith(".pdf"):
            continue
        pdf_urls.add(full_url)

    return sorted(pdf_urls)


def build_section_record(section_url: str, html: str) -> Dict[str, Any]:
    return {
        "section_url": canonicalize_url(section_url),
        "section_slug": Path(urlparse(section_url).path.rstrip("/")).name,
        "title": extract_page_title(html),
        "pdf_urls": extract_pdf_urls(html, section_url),
    }


def get_supabase_client() -> Client:
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
    )

    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Supabase credentials are missing. Set SUPABASE_URL and a Supabase key in the environment."
        )

    return create_client(supabase_url, supabase_key)


def load_pathway_documents(client: Client, page_size: int = 1000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    start = 0

    while True:
        response = (
            client.table(PATHWAY_DOCUMENTS_TABLE)
            .select("pathway_id,doc_name,doc_display_name,doc_file_path,active")
            .eq("active", True)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break

        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return rows


def build_document_indexes(pathway_documents: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    indexes: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "by_doc_name": {},
        "by_pathway_id": {},
        "by_basename": {},
    }

    def add(index_name: str, raw_value: Optional[str], row: Dict[str, Any]) -> None:
        key = normalize_key(raw_value)
        if not key:
            return
        indexes[index_name].setdefault(key, []).append(row)

    for row in pathway_documents:
        add("by_doc_name", row.get("doc_name"), row)
        add("by_pathway_id", row.get("pathway_id"), row)
        add("by_basename", Path(row.get("doc_file_path") or row.get("doc_name") or "").stem, row)

    return indexes


def match_document(
    pdf_url: str, indexes: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Optional[Dict[str, Any]]:
    basename_key = normalize_key(pdf_basename(pdf_url))
    if not basename_key:
        return None

    candidates: List[Dict[str, Any]] = []
    seen = set()

    for index_name in ("by_doc_name", "by_basename", "by_pathway_id"):
        for row in indexes[index_name].get(basename_key, []):
            row_key = (row.get("pathway_id"), row.get("doc_name"))
            if row_key in seen:
                continue
            seen.add(row_key)
            candidates.append(row)

    if not candidates:
        return None

    def match_priority(row: Dict[str, Any]) -> tuple[Any, ...]:
        return (
            0 if normalize_key(row.get("doc_name")) == basename_key else 1,
            0 if normalize_key(Path(row.get("doc_file_path") or "").stem) == basename_key else 1,
            0 if normalize_key(row.get("pathway_id")) == basename_key else 1,
            normalize_key(row.get("doc_display_name") or row.get("doc_name") or row.get("pathway_id")),
        )

    return min(candidates, key=match_priority)


def build_resource_id(pdf_url: str, existing_ids: Iterable[str]) -> str:
    base_id = normalize_key(pdf_basename(pdf_url)) or "resource"
    existing = set(existing_ids)
    if base_id not in existing:
        return base_id

    suffix = 2
    while f"{base_id}-{suffix}" in existing:
        suffix += 1
    return f"{base_id}-{suffix}"


def is_main_resource(resource: Dict[str, str]) -> bool:
    text = " ".join(
        [
            resource.get("id", ""),
            resource.get("label", ""),
            resource.get("doc_name", ""),
        ]
    ).lower()
    if any(token in text for token in PRIMARY_RESOURCE_HINTS):
        return True
    return "pathway" in text and not any(token in text for token in SECONDARY_RESOURCE_HINTS)


def resource_sort_key(resource: Dict[str, str]) -> tuple[Any, ...]:
    text = " ".join(
        [
            resource.get("id", ""),
            resource.get("label", ""),
            resource.get("doc_name", ""),
        ]
    ).lower()
    secondary_penalty = 1 if any(token in text for token in SECONDARY_RESOURCE_HINTS) else 0
    return (
        0 if is_main_resource(resource) else 1,
        secondary_penalty,
        resource.get("label", "").lower(),
        resource.get("id", ""),
    )


def pathway_sort_key(pathway: Dict[str, Any]) -> tuple[Any, ...]:
    legacy_index = {pathway_id: index for index, pathway_id in enumerate(LEGACY_ORDER)}
    if pathway["id"] in legacy_index:
        return (0, legacy_index[pathway["id"]], pathway["label"].lower())
    return (1, pathway["label"].lower(), pathway["id"])


def build_catalog(
    section_records: Sequence[Dict[str, Any]], pathway_documents: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    indexes = build_document_indexes(pathway_documents)
    catalog: List[Dict[str, Any]] = []

    for section in section_records:
        pdf_urls = section.get("pdf_urls") or []
        if not pdf_urls:
            continue

        resource_records = []
        used_resource_ids: List[str] = []

        for pdf_url in sorted(set(pdf_urls)):
            matched = match_document(pdf_url, indexes)
            resource = {
                "id": build_resource_id(pdf_url, used_resource_ids),
                "label": (
                    matched.get("doc_display_name")
                    if matched and matched.get("doc_display_name")
                    else titleize_filename(pdf_basename(pdf_url))
                ),
                "doc_name": (
                    matched.get("doc_name")
                    if matched and matched.get("doc_name")
                    else pdf_basename(pdf_url)
                ),
                "pdf_url": pdf_url,
            }
            if matched and matched.get("pathway_id"):
                resource["medembed_id"] = matched["pathway_id"]
            used_resource_ids.append(resource["id"])
            resource_records.append({"resource": resource, "matched": matched})

        resource_records.sort(key=lambda item: resource_sort_key(item["resource"]))
        resources = [item["resource"] for item in resource_records]
        default_resource = resources[0]

        retrieval_document_ids: List[str] = []
        for item in resource_records:
            matched = item["matched"]
            pathway_id = matched.get("pathway_id") if matched else None
            if pathway_id and pathway_id not in retrieval_document_ids:
                retrieval_document_ids.append(pathway_id)

        section_slug = section.get("section_slug") or Path(urlparse(section["section_url"]).path.rstrip("/")).name
        pathway_id = ID_OVERRIDES.get(section_slug, section_slug)
        label = section.get("title") or titleize_filename(section_slug)

        catalog.append(
            {
                "id": pathway_id,
                "label": label,
                "default_resource_id": default_resource["id"],
                "doc_name": default_resource["doc_name"],
                "pdf_url": default_resource["pdf_url"],
                "retrieval_document_ids": retrieval_document_ids,
                "resources": resources,
            }
        )

    return sorted(catalog, key=pathway_sort_key)


def prepare_catalog_for_render(catalog: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared_catalog: List[Dict[str, Any]] = []

    for pathway in catalog:
        resources = []
        for resource in pathway.get("resources", []):
            if not resource.get("pdf_url"):
                continue
            resources.append(dict(resource))

        if not resources:
            continue

        resource_ids = {resource["id"] for resource in resources}
        default_resource_id = pathway["default_resource_id"]
        if default_resource_id not in resource_ids:
            raise ValueError(
                f"Pathway '{pathway['id']}' default_resource_id '{default_resource_id}' "
                "must reference a visible PDF resource."
            )

        default_resource = next(
            resource for resource in resources if resource["id"] == default_resource_id
        )
        retrieval_document_ids = list(pathway.get("retrieval_document_ids", []))
        for resource in resources:
            medembed_id = resource.get("medembed_id")
            if medembed_id and medembed_id not in retrieval_document_ids:
                retrieval_document_ids.append(medembed_id)

        prepared_catalog.append(
            {
                "id": pathway["id"],
                "label": pathway["label"],
                "default_resource_id": default_resource_id,
                "doc_name": default_resource["doc_name"],
                "pdf_url": default_resource["pdf_url"],
                "retrieval_document_ids": retrieval_document_ids,
                "resources": resources,
            }
        )

    return sorted(prepared_catalog, key=pathway_sort_key)


def apply_resource_pdf_url_overrides(
    catalog: Sequence[Dict[str, Any]],
    resource_pdf_url_overrides: Sequence[Dict[str, str]],
) -> List[Dict[str, Any]]:
    if not resource_pdf_url_overrides:
        return [dict(pathway, resources=[dict(resource) for resource in pathway["resources"]]) for pathway in catalog]

    prepared_catalog: List[Dict[str, Any]] = []
    pathway_by_id = {pathway["id"]: pathway for pathway in catalog}

    for override in resource_pdf_url_overrides:
        pathway_id = override.get("pathway_id", "").strip()
        resource_id = override.get("resource_id", "").strip()
        pdf_url = override.get("pdf_url", "").strip()
        if not pathway_id or not resource_id or not pdf_url:
            raise ValueError(
                "Each resource PDF override must declare pathway_id, resource_id, and pdf_url."
            )
        if pathway_id not in pathway_by_id:
            raise ValueError(f"PDF override references unknown pathway '{pathway_id}'.")

    for pathway in catalog:
        resource_copies = [dict(resource) for resource in pathway["resources"]]
        updated_pathway = dict(pathway, resources=resource_copies)
        for override in resource_pdf_url_overrides:
            if override["pathway_id"] != pathway["id"]:
                continue
            target_resource = next(
                (resource for resource in resource_copies if resource["id"] == override["resource_id"]),
                None,
            )
            if not target_resource:
                raise ValueError(
                    f"PDF override references unknown resource '{override['resource_id']}' "
                    f"in pathway '{pathway['id']}'."
                )
            target_resource["pdf_url"] = override["pdf_url"]
            if updated_pathway["default_resource_id"] == target_resource["id"]:
                updated_pathway["pdf_url"] = override["pdf_url"]
                updated_pathway["doc_name"] = target_resource["doc_name"]
        prepared_catalog.append(updated_pathway)

    return prepared_catalog


def apply_hidden_source_aliases(
    catalog: Sequence[Dict[str, Any]],
    hidden_source_aliases: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    pathway_by_id = {pathway["id"]: pathway for pathway in catalog}
    normalized_aliases: List[Dict[str, str]] = []
    seen_sources = set()

    for alias in hidden_source_aliases:
        pathway_id = alias.get("pathway_id", "").strip()
        source_file = alias.get("source_file", "").strip()
        preview_resource_id = alias.get("preview_resource_id", "").strip()

        if not pathway_id or not source_file or not preview_resource_id:
            raise ValueError(
                "Each hidden source alias must declare pathway_id, source_file, and preview_resource_id."
            )
        if source_file in seen_sources:
            raise ValueError(f"Duplicate hidden source alias for '{source_file}'.")

        pathway = pathway_by_id.get(pathway_id)
        if not pathway:
            raise ValueError(f"Hidden source alias references unknown pathway '{pathway_id}'.")

        preview_resource = next(
            (
                resource
                for resource in pathway["resources"]
                if resource["id"] == preview_resource_id
            ),
            None,
        )
        if not preview_resource:
            raise ValueError(
                f"Hidden source alias '{source_file}' references unknown preview resource "
                f"'{preview_resource_id}' in pathway '{pathway_id}'."
            )

        retrieval_document_ids = pathway.setdefault("retrieval_document_ids", [])
        if source_file not in retrieval_document_ids:
            retrieval_document_ids.append(source_file)

        normalized_aliases.append(
            {
                "pathway_id": pathway_id,
                "source_file": source_file,
                "preview_resource_id": preview_resource_id,
            }
        )
        seen_sources.add(source_file)

    return sorted(
        normalized_aliases,
        key=lambda alias: (alias["pathway_id"], alias["source_file"], alias["preview_resource_id"]),
    )


def render_catalog_module(
    catalog: Sequence[Dict[str, Any]],
    hidden_source_aliases: Sequence[Dict[str, str]],
    active_pathway_ids: Optional[Sequence[str]] = None,
) -> str:
    catalog_literal = pprint.pformat(list(catalog), width=100, sort_dicts=False)
    aliases_literal = pprint.pformat(list(hidden_source_aliases), width=100, sort_dicts=False)
    if active_pathway_ids is None:
        active_literal = "None"
    else:
        active_items = sorted(set(active_pathway_ids))
        active_literal = "{\n" + "".join(f'    "{item}",\n' for item in active_items) + "}"
    return (
        "from pathlib import Path\n"
        "from typing import Any, Dict, List, Optional\n"
        "from urllib.parse import unquote, urlparse\n\n\n"
        "# Generated by generate_pathways_catalog.py. Do not edit manually.\n\n"
        f"PATHWAY_CATALOG: List[Dict[str, Any]] = {catalog_literal}\n\n"
        f"HIDDEN_SOURCE_ALIASES: List[Dict[str, str]] = {aliases_literal}\n\n"
        f"ACTIVE_PATHWAY_IDS = {active_literal}\n\n"
        "def _pdf_basename_from_url(pdf_url: str) -> str:\n"
        "    return unquote(Path(urlparse(pdf_url).path).name)\n\n\n"
        "def _build_document_reference_by_doc_name() -> Dict[str, Dict[str, str]]:\n"
        "    reference_by_doc_name: Dict[str, Dict[str, str]] = {}\n\n"
        "    for pathway in PATHWAY_CATALOG:\n"
        '        for resource in pathway["resources"]:\n'
        '            doc_name = resource.get("doc_name")\n'
        '            pdf_url = resource.get("pdf_url")\n'
        "            if not doc_name or not pdf_url:\n"
        "                continue\n"
        "            reference = {\n"
        '                "pathway_id": pathway["id"],\n'
        '                "resource_id": resource["id"],\n'
        '                "doc_name": doc_name,\n'
        '                "pdf_url": pdf_url,\n'
        '                "pdf_basename": _pdf_basename_from_url(pdf_url),\n'
        "            }\n"
        "            reference_by_doc_name[doc_name] = reference\n"
        '            medembed_id = resource.get("medembed_id")\n'
        "            if medembed_id:\n"
        "                reference_by_doc_name[medembed_id] = reference\n\n"
        "    return reference_by_doc_name\n\n\n"
        "def _build_source_reference_by_key() -> Dict[str, Dict[str, str]]:\n"
        "    reference_by_key: Dict[str, Dict[str, str]] = {}\n"
        "    resource_id_counts: Dict[str, int] = {}\n\n"
        "    for pathway in PATHWAY_CATALOG:\n"
        '        reference_by_key[pathway["id"]] = {\n'
        '            "pathway_id": pathway["id"],\n'
        '            "preview_resource_id": pathway["default_resource_id"],\n'
        "        }\n"
        '        for resource in pathway["resources"]:\n'
        '            resource_id_counts[resource["id"]] = resource_id_counts.get(resource["id"], 0) + 1\n\n'
        "    for pathway in PATHWAY_CATALOG:\n"
        '        for resource in pathway["resources"]:\n'
        "            reference = {\n"
        '                "pathway_id": pathway["id"],\n'
        '                "preview_resource_id": resource["id"],\n'
        "            }\n"
        '            if resource_id_counts.get(resource["id"], 0) == 1:\n'
        '                reference_by_key[resource["id"]] = reference\n'
        '            medembed_id = resource.get("medembed_id")\n'
        "            if medembed_id:\n"
        "                reference_by_key[medembed_id] = reference\n\n"
        "    for alias in HIDDEN_SOURCE_ALIASES:\n"
        '        reference_by_key[alias["source_file"]] = {\n'
        '            "pathway_id": alias["pathway_id"],\n'
        '            "preview_resource_id": alias["preview_resource_id"],\n'
        "        }\n\n"
        "    return reference_by_key\n\n\n"
        'PATHWAY_BY_ID: Dict[str, Dict[str, str]] = {\n'
        '    pathway["id"]: pathway for pathway in PATHWAY_CATALOG\n'
        "}\n\n\n"
        "DOCUMENT_REFERENCE_BY_DOC_NAME = _build_document_reference_by_doc_name()\n\n\n"
        "SOURCE_REFERENCE_BY_KEY = _build_source_reference_by_key()\n\n\n"
        "def list_pathways() -> List[Dict[str, str]]:\n"
        "    if ACTIVE_PATHWAY_IDS is None:\n"
        "        return PATHWAY_CATALOG\n"
        '    return [pathway for pathway in PATHWAY_CATALOG if pathway["id"] in ACTIVE_PATHWAY_IDS]\n\n\n'
        "def get_pathway_by_id(pathway_id: str) -> Optional[Dict[str, str]]:\n"
        "    return PATHWAY_BY_ID.get(pathway_id)\n\n\n"
        "def get_pathway_resource(pathway_id: str, resource_id: Optional[str] = None) -> Optional[Dict[str, str]]:\n"
        "    pathway = get_pathway_by_id(pathway_id)\n"
        "    if not pathway:\n"
        "        return None\n\n"
        '    desired_resource_id = resource_id or pathway["default_resource_id"]\n'
        '    for resource in pathway["resources"]:\n'
        '        if resource["id"] == desired_resource_id:\n'
        "            return resource\n"
        "    return None\n\n\n"
        "def get_pathway_retrieval_documents(pathway_id: str) -> List[str]:\n"
        "    pathway = get_pathway_by_id(pathway_id)\n"
        "    if not pathway:\n"
        "        return []\n"
        '    return list(pathway.get("retrieval_document_ids", []))\n\n\n'
        "def resolve_document_reference(doc_name: Optional[str]) -> Optional[Dict[str, str]]:\n"
        "    if not doc_name:\n"
        "        return None\n"
        "    reference = DOCUMENT_REFERENCE_BY_DOC_NAME.get(doc_name)\n"
        "    return dict(reference) if reference else None\n\n\n"
        "def resolve_source_reference(source_key: Optional[str]) -> Optional[Dict[str, str]]:\n"
        "    if not source_key:\n"
        "        return None\n"
        "    reference = SOURCE_REFERENCE_BY_KEY.get(source_key)\n"
        "    return dict(reference) if reference else None\n"
    )


def fetch_section_records(source_url: str) -> List[Dict[str, Any]]:
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(20.0, connect=10.0),
    ) as client:
        landing_response = client.get(source_url)
        landing_response.raise_for_status()
        section_urls = extract_section_links(landing_response.text, source_url)

        records = []
        for section_url in section_urls:
            section_response = client.get(section_url)
            section_response.raise_for_status()
            records.append(build_section_record(section_url, section_response.text))
        return records


def write_catalog_module(output_path: Path, module_text: str) -> None:
    output_path.write_text(module_text, encoding="utf-8")


def generate_catalog_module(source_url: str, output_path: Path) -> List[Dict[str, Any]]:
    section_records = fetch_section_records(source_url)
    client = get_supabase_client()
    pathway_documents = load_pathway_documents(client)
    catalog = apply_resource_pdf_url_overrides(
        prepare_catalog_for_render(build_catalog(section_records, pathway_documents)),
        RESOURCE_PDF_URL_OVERRIDES,
    )
    hidden_source_aliases = apply_hidden_source_aliases(catalog, HIDDEN_SOURCE_ALIASES)
    module_text = render_catalog_module(catalog, hidden_source_aliases, ACTIVE_PATHWAY_IDS)
    write_catalog_module(output_path, module_text)
    return catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the backend pathways catalog.")
    parser.add_argument("--source-url", default=START_URL, help="Clinical pathways landing page URL.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write the generated Python catalog module.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    catalog = generate_catalog_module(args.source_url, output_path)
    print(f"Generated {len(catalog)} pathways -> {output_path}")


if __name__ == "__main__":
    main()
