create or replace function public.match_chunks_medembed_large(
  query_embedding vector(1024),
  match_count int default 5,
  filter_pathway_tag varchar default null
)
returns table (
  chunk_id bigint,
  chunk_text text,
  chunk_length integer,
  source_docs text[],
  pathway_tag varchar,
  pathway_id varchar,
  similarity double precision
)
language sql
stable
as $$
  select
    pc.chunk_id,
    pc.chunk_text,
    pc.chunk_length,
    pc.source_docs,
    pc.pathway_tag,
    pc.pathway_id,
    1 - (pc.embedding <=> query_embedding) as similarity
  from public.pathway_chunks_medembed_large pc
  where filter_pathway_tag is null or pc.pathway_tag = filter_pathway_tag
  order by pc.embedding <=> query_embedding
  limit greatest(coalesce(match_count, 5), 1);
$$;

grant execute on function public.match_chunks_medembed_large(vector, int, varchar)
to anon, authenticated, service_role;
