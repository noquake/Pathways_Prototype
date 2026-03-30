from typing import Any, Dict, List, Optional
from .models import EMBEDDING_MODELS, DEFAULT_MODEL
from .embeddings import get_embeddings


def _extract_source_docs(row: Dict[str, Any]) -> List[str]:
    raw_source_docs = row.get("source_docs")
    if isinstance(raw_source_docs, list):
        return [str(item).strip() for item in raw_source_docs if str(item).strip()]
    if isinstance(raw_source_docs, str) and raw_source_docs.strip():
        return [raw_source_docs.strip()]
    return []


def _candidate_lookup_tables(config: Dict[str, Any]) -> List[str]:
    table_name = config.get("table")
    if not table_name:
        return []

    tables = [table_name]
    if table_name == "pathway_chunks_linilm":
        tables.append("pathway_chunks")
    return tables


def _hydrate_missing_chunk_metadata(
    supabase,
    results: List[Dict[str, Any]],
    lookup_tables: List[str],
) -> List[Dict[str, Any]]:
    hydrated_results = [dict(item) for item in results]
    missing_chunk_ids = []

    for item in hydrated_results:
        source_docs = _extract_source_docs(item)
        has_pathway_id = bool(str(item.get("pathway_id") or "").strip())
        has_pathway_tag = bool(str(item.get("pathway_tag") or "").strip())
        if source_docs and has_pathway_id and has_pathway_tag:
            item["source_docs"] = source_docs
            continue

        chunk_id = item.get("chunk_id")
        if chunk_id is not None:
            missing_chunk_ids.append(str(chunk_id))

    if not missing_chunk_ids or not lookup_tables:
        return hydrated_results

    metadata_by_chunk_id: Dict[str, Dict[str, Any]] = {}
    for table_name in lookup_tables:
        try:
            response = (
                supabase.table(table_name)
                .select("chunk_id,source_docs,pathway_id,pathway_tag")
                .in_("chunk_id", missing_chunk_ids)
                .execute()
            )
        except Exception:
            continue

        for row in response.data or []:
            source_docs = _extract_source_docs(row)
            metadata_by_chunk_id[str(row.get("chunk_id"))] = {
                "source_docs": source_docs,
                "pathway_id": row.get("pathway_id"),
                "pathway_tag": row.get("pathway_tag"),
            }

        if len(metadata_by_chunk_id) == len(set(missing_chunk_ids)):
            break

    for item in hydrated_results:
        metadata = metadata_by_chunk_id.get(str(item.get("chunk_id")))
        if not metadata:
            continue
        hydrated_fields = []
        if not _extract_source_docs(item) and metadata.get("source_docs"):
            item["source_docs"] = metadata["source_docs"]
            hydrated_fields.append("source_docs")
        if not str(item.get("pathway_id") or "").strip() and metadata.get("pathway_id") is not None:
            item["pathway_id"] = metadata["pathway_id"]
            hydrated_fields.append("pathway_id")
        if not str(item.get("pathway_tag") or "").strip() and metadata.get("pathway_tag") is not None:
            item["pathway_tag"] = metadata["pathway_tag"]
            hydrated_fields.append("pathway_tag")
        if hydrated_fields:
            item["_hydrated_fields"] = hydrated_fields

    return hydrated_results

# def retrieve_chunks(supabase, query: str, top_k: int = 5, pathway_id: Optional[str] = None, pathway_tag: Optional[str] = None, model_key: str = DEFAULT_MODEL):
def retrieve_chunks(
    supabase,
    query: str,
    top_k: int = 5,
    pathway_id: Optional[str] = None,
    pathway_ids: Optional[List[str]] = None,
    model_key: str = DEFAULT_MODEL,
    rpc_function: Optional[str] = None,
    filter_arg: Optional[str] = None,
):
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    config = EMBEDDING_MODELS[model_key]
    rpc_name = rpc_function or config["rpc_function"]
    lookup_tables = _candidate_lookup_tables(config)
    # Explicit filter_arg takes precedence over model config (e.g. doc-scoped queries)
    filter_arg = filter_arg or config.get("filter_arg", "filter_pathway_id")

    # is_query=True ensures MedCPT uses its Query Encoder here
    query_emb = get_embeddings([query], model_key=model_key, is_query=True)[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    payload = {
        "query_embedding": query_emb_list,
        "match_count": top_k,
    }

    # if pathway_id:
    #     payload["filter_pathway_id"] = pathway_id
    # if pathway_tag:
    #     payload["filter_pathway_tag"] = pathway_tag

    try:
        if pathway_ids:
            merged_results = []
            for candidate_pathway_id in pathway_ids:
                candidate_payload = dict(payload)
                candidate_payload[filter_arg] = candidate_pathway_id
                response = supabase.rpc(rpc_name, candidate_payload).execute()
                merged_results.extend(response.data or [])

            merged_results.sort(
                key=lambda item: float(item.get("similarity", item.get("distance", 0.0))),
                reverse=True,
            )
            return _hydrate_missing_chunk_metadata(
                supabase,
                merged_results[:top_k],
                lookup_tables,
            )

        if pathway_id:
            payload[filter_arg] = pathway_id

        response = supabase.rpc(rpc_name, payload).execute()
        return _hydrate_missing_chunk_metadata(
            supabase,
            response.data or [],
            lookup_tables,
        )
    except Exception as e:
        raise RuntimeError(f"Retrieval failed for {model_key}: {e}")
