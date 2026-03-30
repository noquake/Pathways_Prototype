-- Adds a pathway-aware overload for the existing RPC:
--   public.match_pathway_chunks(query_embedding vector(384), match_count int)
--
-- After running this, backend calls that include `filter_pathway_id`
-- will be filtered in-database instead of client-side post-filtering.

create or replace function public.match_medcpt_pathway_chunks(

  query_embedding vector(1024),
  match_count int default 5,
  filter_pathway_id varchar default null
)
returns table (
  chunk_id bigint,
  chunk_text text,
  chunk_length integer,
  source_docs text[],
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
    pc.pathway_id,
    1 - (pc.embedding <=> query_embedding) as similarity
 
  from public.pathway_chunks_medcpt pc

  where filter_pathway_id is null or pc.pathway_id = filter_pathway_id
  order by pc.embedding <=> query_embedding
  limit greatest(coalesce(match_count, 5), 1);
$$;

grant execute on function public.match_medcpt_pathway_chunks(vector, int, varchar)

to anon, authenticated, service_role;
