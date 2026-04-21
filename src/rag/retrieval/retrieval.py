from src.services.llm import openAI
from fastapi import HTTPException
from src.services.supabase import supabase
from src.rag.retrieval.utils import (
    get_project_settings,
    get_project_document_ids,
    build_context_from_retrieved_chunks,
    generate_query_variations,
)
from typing import List, Dict
from src.rag.retrieval.utils import rrf_rank_and_fuse


def retrieve_context(project_id, user_query):
    print("call retrieve_context")

    try:
        """
        RAG Retrieval 파이프라인 단계:
        * Step 1: 데이터베이스에서 사용자 프로젝트 설정 조회
        * Step 2: 현재 프로젝트의 document ID 목록 조회
        * Step 3: RPC 함수를 사용한 벡터 검색으로 가장 관련성 높은 청크 탐색
        * Step 4: RPC 함수를 사용한 하이브리드 검색 (벡터 + 키워드 검색 결합)
        * Step 5: 멀티쿼리 벡터 검색 (여러 쿼리 변형 생성 후 검색)
        * Step 6: 멀티쿼리 하이브리드 검색 (여러 쿼리로 hybrid 전략 적용)
        * Step 7: 검색된 청크에서 컨텍스트 구성 및 인용 포함 구조화된 컨텍스트로 포맷
        """
        # Step 1: 데이터베이스에서 사용자 프로젝트 설정 조회
        project_settings = get_project_settings(project_id)

        # Step 2: 현재 프로젝트의 document ID 목록 조회
        document_ids = get_project_document_ids(project_id)
        # print("Found document IDs: ", len(document_ids))

        # Step 4 & 5: 선택된 전략에 따라 검색 실행
        strategy = project_settings["rag_strategy"]
        chunks = []
        if strategy == "basic":
            # Basic RAG 전략: 벡터 검색만 수행
            chunks = vector_search(user_query, document_ids, project_settings)
            print(f"Vector search resulted in: {len(chunks)} chunks")

        elif strategy == "hybrid":
            # Hybrid RAG 전략: 벡터 + 키워드 검색을 RRF 랭킹으로 결합
            chunks = hybrid_search(user_query, document_ids, project_settings)
            print(f"Hybrid search resulted in: {len(chunks)} chunks")

        # Step 6: 멀티쿼리 벡터 검색
        elif strategy == "multi-query-vector":
            chunks = multi_query_vector_search(
                user_query, document_ids, project_settings
            )
            print(f"Multi-query vector search resulted in: {len(chunks)} chunks")

        # Step 7: 멀티쿼리 하이브리드 검색
        elif strategy == "multi-query-hybrid":
            chunks = multi_query_hybrid_search(
                user_query, document_ids, project_settings
            )
            print(f"Multi-query hybrid search resulted in: {len(chunks)} chunks")

        # Step 8: 상위 k개 청크 선택
        chunks = chunks[: project_settings["final_context_size"]]

        # Step 9: 검색된 청크에서 컨텍스트 구성 및 인용 포함 구조화된 컨텍스트로 포맷
        texts, images, tables, citations = build_context_from_retrieved_chunks(chunks)
        # validate_context_from_retrieved_chunks(texts, images, tables, citations)

        return texts, images, tables, citations
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed in RAG's Retrieval: {str(e)}"
        )


def vector_search(user_query, document_ids, project_settings):
    user_query_embedding = openAI["embeddings"].embed_documents([user_query])[0]
    vector_search_result_chunks = supabase.rpc(
        "vector_search_document_chunks",
        {
            "query_embedding": user_query_embedding,
            "filter_document_ids": document_ids,
            "match_threshold": project_settings["similarity_threshold"],
            "chunks_per_search": project_settings["chunks_per_search"],
        },
    ).execute()
    return vector_search_result_chunks.data if vector_search_result_chunks.data else []


def keyword_search(query, document_ids, settings):
    keyword_search_result_chunks = supabase.rpc(
        "keyword_search_document_chunks",
        {
            "query_text": query,
            "filter_document_ids": document_ids,
            "chunks_per_search": settings["chunks_per_search"],
        },
    ).execute()

    return (
        keyword_search_result_chunks.data if keyword_search_result_chunks.data else []
    )


def hybrid_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    """벡터와 키워드 검색 결과를 결합하는 hybrid 검색 실행"""
    # 두 검색 방법의 결과 조회
    vector_results = vector_search(query, document_ids, settings)
    keyword_results = keyword_search(query, document_ids, settings)

    print(f"📈 Vector search returned: {len(vector_results)} chunks")
    print(f"📈 Keyword search returned: {len(keyword_results)} chunks")

    # 설정된 가중치로 RRF를 사용하여 결합
    return rrf_rank_and_fuse(
        [vector_results, keyword_results],
        [settings["vector_weight"], settings["keyword_weight"]],
    )


def multi_query_vector_search(user_query, document_ids, project_settings):
    """쿼리 변형을 사용하는 멀티쿼리 벡터 검색 실행"""
    queries = generate_query_variations(
        user_query, project_settings["number_of_queries"]
    )
    print(f"Generated {len(queries)} query variations")

    all_chunks = []
    for index, query in enumerate(queries):
        chunks = vector_search(query, document_ids, project_settings)
        all_chunks.append(chunks)
        print(
            f"Vector search for query {index+1}/{len(queries)}: {query} resulted in: {len(chunks)} chunks"
        )

    final_chunks = rrf_rank_and_fuse(all_chunks)
    print(f"RRF Fusion returned {len(final_chunks)} chunks")
    return final_chunks


def multi_query_hybrid_search(user_query, document_ids, project_settings):
    """쿼리 변형을 사용하는 멀티쿼리 hybrid 검색 실행"""
    queries = generate_query_variations(
        user_query, project_settings["number_of_queries"]
    )
    print(f"Generated {len(queries)} query variations for hybrid search")

    all_chunks = []
    for index, query in enumerate(queries):
        chunks = hybrid_search(query, document_ids, project_settings)
        all_chunks.append(chunks)
        print(
            f"Hybrid search for query {index+1}/{len(queries)}: {query} resulted in: {len(chunks)} chunks"
        )

    final_chunks = rrf_rank_and_fuse(all_chunks)
    print(f"RRF Fusion returned {len(final_chunks)} chunks")
    return final_chunks