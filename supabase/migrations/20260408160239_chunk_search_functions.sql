CREATE OR REPLACE FUNCTION vector_search_document_chunks(
    query_embedding vector,
    filter_document_ids uuid[],
    match_threshold double precision DEFAULT 0.3,
    chunks_per_search integer DEFAULT 20
)
RETURNS TABLE(
    id uuid,
    document_id uuid,
    content text,
    chunk_index integer,
    created_at timestamp with time zone,
    page_number integer,
    char_count integer,
    type jsonb,
    original_content jsonb,
    embedding vector
)
LANGUAGE sql
AS $function$
SELECT
    dc.id,
    dc.document_id,
    dc.content,
    dc.chunk_index,
    dc.created_at,
    dc.page_number,
    dc.char_count,
    dc.type,
    dc.original_content,
    dc.embedding
FROM
    document_chunks dc
WHERE 
    dc.document_id = ANY(filter_document_ids)
    AND dc.embedding IS NOT NULL
    AND (1 - (dc.embedding <=> query_embedding)) > match_threshold  
    -- 1. <=>: cosine distance operator (pgvector), Lower value = more similar (vectors are closer)
    -- 2. 1 - (): convert cosine distance into a cosine similarity (higher value = more similar)
ORDER BY
    dc.embedding <=> query_embedding ASC
LIMIT 
    chunks_per_search;
$function$;
      
-- <=> 연산을 where절과 orderby 절에서 두 번 사용함 
-- postgreSQL은 두 연산에서 HNSW index를 타기 때문에 모두 빠르게 처리할 수 있다. 
