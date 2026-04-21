from src.services.supabase import supabase
from fastapi import HTTPException
from typing import List, Dict, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from src.services.llm import openAI
from src.models.schemas import QueryVariations


def get_project_settings(project_id):
    try:
        project_settings_result = (
            supabase.table("project_settings")
            .select("*")
            .eq("project_id", project_id)
            .execute()
        )

        if not project_settings_result.data:
            raise HTTPException(status_code=404, detail="Project settings not found")

        project_settings = project_settings_result.data[0]
        return project_settings
    except Exception as e:
        raise Exception(f"Failed to get project settings: {str(e)}")


def get_project_document_ids(project_id):
    try:
        document_ids_result = (
            supabase.table("project_documents")
            .select("id")
            .eq("project_id", project_id)
            .execute()
        )

        if not document_ids_result.data:
            return []

        document_ids = [document["id"] for document in document_ids_result.data]
        return document_ids
    except Exception as e:
        raise Exception(f"Failed to get document IDs: {str(e)}")


def build_context_from_retrieved_chunks(
    chunks: List[Dict],
) -> Tuple[List[str], List[str], List[str], List[Dict]]:
    """
    검색된 청크에서 컨텍스트를 구성하고, 인용 정보가 포함된 구조화된 컨텍스트로 포맷합니다.
    인용은 문서 정보와 청크의 페이지 번호를 담고 있는 citations 목록의 항목입니다.
    """
    if not chunks:
        return [], [], [], []

    texts = []
    images = []
    tables = []
    citations = []

    # 모든 청크의 파일명을 단일 쿼리로 일괄 조회
    doc_ids = [chunk["document_id"] for chunk in chunks if chunk.get("document_id")]
    # doc_ids 목록에서 중복 없는 고유 document ID 추출
    unique_doc_ids = list(set(doc_ids))

    # unique_doc_ids에 포함된 문서들의 파일명을 저장할 딕셔너리 생성
    filename_map = {}

    # unique_doc_ids에 포함된 문서들의 파일명 조회
    if unique_doc_ids:
        result = (
            supabase.table("project_documents")
            .select("id, filename")
            .in_("id", unique_doc_ids)
            .execute()
        )
        filename_map = {doc["id"]: doc["filename"] for doc in result.data}

    # 각 청크 처리
    for chunk in chunks:
        original_content = chunk.get("original_content", {})

        # 청크에서 콘텐츠 추출
        chunk_text = original_content.get("text", "")
        chunk_images = original_content.get("images", [])
        chunk_tables = original_content.get("tables", [])

        if (
            chunk_text
        ):  # chunk_text는 배열이 아니므로 append로 추가
            texts.append(chunk_text)
        # chunk_images와 chunk_tables는 배열이므로 extend로 추가
        images.extend(chunk_images)
        tables.extend(chunk_tables)

        # * 각 청크마다 인용 정보 추가
        doc_id = chunk.get("document_id")
        if doc_id:
            citations.append(
                {
                    "chunk_id": chunk.get("id"),
                    "document_id": doc_id,
                    "filename": filename_map.get(doc_id, "Unknown Document"),
                    "page": chunk.get("page_number", "Unknown"),
                }
            )

    return texts, images, tables, citations


def validate_context_from_retrieved_chunks(
    texts: List[str], images: List[str], tables: List[str], citations: List[Dict]
) -> None:
    """검색된 청크의 컨텍스트 데이터를 검증하고 읽기 좋은 형식으로 출력"""
    print("\n" + "=" * 80)
    print("📦 CONTEXT VALIDATION")
    print("=" * 80)

    # 텍스트 - 전체 텍스트 출력
    print(f"\n📝 TEXTS: {len(texts)} chunks")
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*80}")
        print(f"CHUNK [{i}] - {len(text)} characters")
        print(f"{'='*80}")
        print(text)
        print(f"{'='*80}\n")

    # 이미지
    print(f"\n🖼️  IMAGES: {len(images)}")
    for i, img in enumerate(images, 1):
        img_preview = str(img)[:60] + ("..." if len(str(img)) > 60 else "")
        print(f"  [{i}] {img_preview}")

    # 테이블
    print(f"\n📊 TABLES: {len(tables)}")
    for i, table in enumerate(tables, 1):
        if isinstance(table, dict):
            rows = len(table.get("rows", []))
            cols = len(table.get("headers", []))
            print(f"  [{i}] {rows} rows × {cols} cols")
        else:
            print(f"  [{i}] Type: {type(table).__name__}")

    # 인용
    print(f"\n📚 CITATIONS: {len(citations)}")
    for i, cite in enumerate(citations, 1):
        chunk_id = cite["chunk_id"][:8] if cite.get("chunk_id") else "N/A"
        print(f"  [{i}] {cite['filename']} (pg.{cite['page']}) | chunk: {chunk_id}...")

    # 요약
    total_chars = sum(len(text) for text in texts)
    print(f"\n{'='*80}")
    print(
        f"✅ Total: {len(texts)} texts ({total_chars:,} chars), {len(images)} images, {len(tables)} tables, {len(citations)} citations"
    )
    print("=" * 80 + "\n")


def prepare_prompt_and_invoke_llm(
    user_query: str, texts: List[str], images: List[str], tables: List[str]
) -> str:
    """
    컨텍스트로 system prompt를 구성하고 멀티모달을 지원하는 LLM을 호출합니다.
    """
    # system prompt 구성 요소 빌드
    prompt_parts = []

    # 주요 지시사항
    prompt_parts.append(
        "You are a helpful AI assistant that answers questions based solely on the provided context. "
        "Your task is to provide accurate, detailed answers using ONLY the information available in the context below.\n\n"
        "IMPORTANT RULES:\n"
        "- Only answer based on the provided context (texts, tables, and images)\n"
        "- If the answer cannot be found in the context, respond with: 'I don't have enough information in the provided context to answer that question.'\n"
        "- Do not use external knowledge or make assumptions beyond what's explicitly stated\n"
        "- When referencing information, be specific and cite relevant parts of the context\n"
        "- Synthesize information from texts, tables, and images to provide comprehensive answers\n\n"
    )

    # 텍스트 컨텍스트 추가
    if texts:
        prompt_parts.append("=" * 80)
        prompt_parts.append("CONTEXT DOCUMENTS")
        prompt_parts.append("=" * 80 + "\n")

        for i, text in enumerate(texts, 1):
            prompt_parts.append(f"--- Document Chunk {i} ---")
            prompt_parts.append(text.strip())
            prompt_parts.append("")

    # 테이블이 있는 경우 추가
    if tables:
        prompt_parts.append("\n" + "=" * 80)
        prompt_parts.append("RELATED TABLES")
        prompt_parts.append("=" * 80)
        prompt_parts.append(
            "The following tables contain structured data that may be relevant to your answer. "
            "Analyze the table contents carefully.\n"
        )

        for i, table_html in enumerate(tables, 1):
            prompt_parts.append(f"--- Table {i} ---")
            prompt_parts.append(table_html)
            prompt_parts.append("")

    # 이미지가 있는 경우 참조 추가
    if images:
        prompt_parts.append("\n" + "=" * 80)
        prompt_parts.append("RELATED IMAGES")
        prompt_parts.append("=" * 80)
        prompt_parts.append(
            f"{len(images)} image(s) will be provided alongside the user's question. "
            "These images may contain diagrams, charts, figures, formulas, or other visual information. "
            "Carefully analyze the visual content when formulating your response. "
            "The images are part of the retrieved context and should be used to answer the question.\n"
        )

    # 최종 지시사항
    prompt_parts.append("=" * 80)
    prompt_parts.append(
        "Based on all the context provided above (documents, tables, and images), "
        "please answer the user's question accurately and comprehensively."
    )
    prompt_parts.append("=" * 80)

    system_prompt = "\n".join(prompt_parts)

    # LLM 메시지 구성
    messages = [SystemMessage(content=system_prompt)]

    # 사용자 쿼리와 이미지를 포함한 사용자 메시지 생성
    if images:
        # 멀티모달 메시지: 텍스트 + 이미지
        content_parts = [{"type": "text", "text": user_query}]

        # 각 이미지를 content 배열에 추가
        for img_base64 in images:
            # data URI 접두사가 있는 경우 base64 문자열 정리
            if img_base64.startswith("data:image"):
                img_base64 = img_base64.split(",", 1)[1]

            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                }
            )

        messages.append(HumanMessage(content=content_parts))
    else:
        # 텍스트 전용 메시지
        messages.append(HumanMessage(content=user_query))

    # LLM 호출 및 응답 반환
    print(
        f"🤖 Invoking LLM with {len(messages)} messages ({len(texts)} texts, {len(tables)} tables, {len(images)} images)..."
    )
    response = openAI["chat_llm"].invoke(messages)

    return response.content


def rrf_rank_and_fuse(search_results_list, weights=None, k=60):
    """RRF (Reciprocal Rank Fusion) 랭킹"""
    if not search_results_list or not any(search_results_list):
        return []

    if weights is None:
        weights = [1.0 / len(search_results_list)] * len(search_results_list)

    chunk_scores = {}
    all_chunks = {}

    for search_idx, results in enumerate(search_results_list):
        weight = weights[search_idx]

        for rank, chunk in enumerate(results):
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue

            rrf_score = weight * (1.0 / (k + rank + 1))

            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] += rrf_score
            else:
                chunk_scores[chunk_id] = rrf_score
                all_chunks[chunk_id] = chunk

    sorted_chunk_ids = sorted(
        chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True
    )
    return [all_chunks[chunk_id] for chunk_id in sorted_chunk_ids]


def generate_query_variations(original_query: str, num_queries: int = 3) -> List[str]:
    """LLM을 사용하여 쿼리 변형 생성"""
    system_prompt = f"""Generate {num_queries-1} alternative ways to phrase this question for document search. Use different keywords and synonyms while maintaining the same intent. Return exactly {num_queries-1} variations."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original query: {original_query}"),
        ]

        structured_llm = openAI["chat_llm"].with_structured_output(QueryVariations)
        result = structured_llm.invoke(messages)

        print(f"✅ Generated {len(result.queries)} query variations")  # ✅ 디버그
        print(f"Queries: {result.queries}")  # ✅ 디버그

        return [original_query] + result.queries[: num_queries - 1]
    except Exception as e:
        print(f"❌ Query variation generation failed: {str(e)}")  # ✅ 더 나은 오류 처리
        import traceback

        traceback.print_exc()  # ✅ 전체 스택 트레이스
        return [original_query]