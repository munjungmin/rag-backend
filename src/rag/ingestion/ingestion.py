from src.services.supabase import supabase
import os
import time
from src.services.llm import openAI
from src.services.awsS3 import s3_client
from src.config.appConfig import appConfig
from src.rag.ingestion.utils import (
    partition_document,
    analyze_elements,
    separate_content_types,
    get_page_number,
    create_ai_summary,
)
from src.models.schemas import ProcessingStatus
#from unstructured.chunking.title import chunk_by_title
from src.services.webScrapper import scrapingbee_client


def process_document(document_id: str):
    """
    * Step 1 : S3에서 파일 다운로드(file) 또는 URL 크롤링(url)하여 PDF에서 텍스트, 테이블, 이미지 추출 (Unstructured 라이브러리 사용)
    * Step 2 : 추출된 콘텐츠를 청크로 분할
    * Step 3 : 각 청크에 대한 AI 요약 생성
    * Step 4 : 청크의 벡터 임베딩 생성 후 PostgreSQL에 저장
    * 필요에 따라 processing_status와 processing_details로 project document 레코드 업데이트
    *   - `processing_details` : UI에 표시할 문서에서 추출한 element 유형 또는 메타데이터
    """

    try:
        update_db_status(document_id, ProcessingStatus.PROCESSING)

        document_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", document_id)
            .execute()
        )

        if not document_result.data:
            raise Exception(
                f"Failed to get project document record with id: {document_id}"
            )
        
        document = document_result.data[0]

        # Step 1 : S3에서 파일 다운로드(file) 또는 URL 크롤링(url)하여 콘텐츠 추출
        update_db_status(document_id, ProcessingStatus.PARTITIONING)

        elements_summary, elements = download_content_and_partition(
            document_id, document
        )

        update_db_status(
            document_id,
            ProcessingStatus.CHUNKING,
            {
                # UI에 표시하기 위해 파티셔닝 결과 저장
                ProcessingStatus.PARTITIONING.value: {
                    "elements_found": elements_summary,
                }
            },
        )

        # Step 2 : 추출된 콘텐츠를 청크로 분할
        chunks, chunking_metrics = chunk_elements_by_title(elements)
        update_db_status(
            document_id,
            ProcessingStatus.SUMMARISING,
            {
                # UI에 표시하기 위해 청킹 결과 저장
                ProcessingStatus.CHUNKING.value: chunking_metrics,
            },
        )

        # Step 3 : 이미지나 테이블이 포함된 청크에 대한 AI 요약 생성
        processed_chunks = summarise_chunks(chunks, document_id)
        update_db_status(document_id, ProcessingStatus.VECTORIZATION)

        # Step 4 : 벡터 임베딩 생성 (1536차원)
        vectorize_chunks_summary_and_store_in_database(processed_chunks, document_id)

        update_db_status(document_id, ProcessingStatus.COMPLETED)

        return {
            "success": True,
            "document_id": document_id,
        }
    except Exception as e:
        raise Exception(f"Failed to process document {document_id}: {str(e)}")


def update_db_status(
    document_id: str, status: ProcessingStatus, details: dict = None
):
    """
    새로운 상태와 세부 정보로 project document 레코드 업데이트
    """
    try:
        # project document 레코드 조회
        document_result = (
            supabase.table("project_documents")
            .select("processing_details")
            .eq("id", document_id)
            .execute()
        )
        if not document_result.data:
            raise Exception(
                f"Failed to get project document record with id: {document_id}"
            )

        # processing details가 있는 경우 project document 레코드에 추가
        current_details = {}
        if document_result.data[0]["processing_details"]:
            current_details = document_result.data[0]["processing_details"]

        # 제공된 경우 새 세부 정보 추가
        if details:
            current_details.update(
                details
            )  # 참고: update() - 다른 딕셔너리를 현재 딕셔너리에 병합하는 내장 dict 메서드

        # 새 세부 정보로 project document 레코드 업데이트
        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "processing_status": status.value,
                    "processing_details": current_details,
                }
            )
            .eq("id", document_id)
            .execute()
        )

        if not document_update_result.data:
            raise Exception(
                f"Failed to update project document record with id: {document_id}"
            )

    except Exception as e:
        raise Exception(f"Failed to update status in database: {str(e)}")


def download_content_and_partition(document_id: str, document: dict):
    """
    document: file 또는 URL
    if : 문서 - S3에서 다운로드
    else : URL - URL 크롤링
    텍스트, 테이블, 이미지 등의 element로 파티셔닝하고 요약을 분석하여 DB에 저장
    """
    try:
        # Get the project document record
        document_source_type = document["source_type"]
        elements = None
        temp_file_path = None

        if document_source_type == "file":
            # S3에서 파일 다운로드
            s3_key = document["s3_key"]
            filename = document["filename"]
            file_type = filename.split(".")[-1].lower()

            # 임시 디렉토리에 파일 다운로드 - 모든 OS 지원 (Linux, Windows, Mac)
            temp_file_path = f"/tmp/{document_id}.{file_type}"
            s3_client.download_file(appConfig["s3_bucket_name"], s3_key, temp_file_path)

            elements = partition_document(temp_file_path, file_type)

        if document_source_type == "url":
            url = document["source_url"]
            # URL 크롤링
            response = scrapingbee_client.get(url)
            temp_file_path = f"/tmp/{document_id}.html"
            with open(temp_file_path, "wb") as f:
                f.write(response.content)

            elements = partition_document(temp_file_path, "html", source_type="url")

        elements_summary = analyze_elements(elements)

        # 임시 파일 삭제
        os.remove(temp_file_path)

        return elements_summary, elements

    except Exception as e:
        raise Exception(
            f"Failed in Step 1 to download content and partition elements: {str(e)}"
        )


def chunk_elements_by_title(elements):
    from unstructured.chunking.title import chunk_by_title
    try:
        chunks = chunk_by_title(
            elements,  # 이전 단계에서 파싱된 PDF elements
            max_characters=3000,  # 최대 한도 - 청크당 3000자 절대 초과 불가
            new_after_n_chars=2400,  # 2400자 이후 새 청크 시작 권장
            combine_text_under_n_chars=500,  # 500자 미만의 작은 청크는 인접 청크와 병합
        )

        # 청킹 메트릭 수집
        total_chunks = len(chunks)

        chunking_metrics = {"total_chunks": total_chunks}

        return chunks, chunking_metrics
    except Exception as e:
        raise Exception(f"Failed to chunk elements by title: {str(e)}")


def summarise_chunks(chunks, document_id, source_type="file"):
    """
    각 청크에 대해 AI 요약을 생성
    각 청크 처리에 최소 5초가 소요되므로 UI를 업데이트하여 UX를 개선
    """

    try:
        processed_chunks = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            current_chunk = i + 1

            # UI 폴링 루프를 위한 진행 상황 업데이트; 사용자에게 진행 상태 알림
            update_db_status(
                document_id,
                ProcessingStatus.SUMMARISING,
                {
                    ProcessingStatus.SUMMARISING.value: {
                        "current_chunk": current_chunk,
                        "total_chunks": total_chunks,
                    },
                },
            )

            # 원본 청크를 유형별 콘텐츠 버킷(텍스트/테이블/이미지 등)으로 정규화
            # content_data = {
            #     "text": "This is the main text content of the chunk...",
            #     "tables": ["<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>"],
            #     "images": ["iVBORw0KGgoAAAANSUhEUgAA..."],  # base64 encoded image strings
            #     "types": ["text", "table", "image"]  # or ["text"], ["text", "table"], etc.
            # }
            content_data = separate_content_types(chunk, source_type)

            # 청크에 테이블 또는 이미지가 하나 이상 있을 때만 AI 요약 사용 (테이블과 이미지는 텍스트 임베딩이 불가능하니까, 임베딩 가능한 형태로 만드는 과정)
            if content_data["tables"] or content_data["images"]:
                enhanced_content = create_ai_summary(
                    content_data["text"], content_data["tables"], content_data["images"]
                )
            else:
                enhanced_content = content_data["text"]

            # UI에서 추적 가능하도록 원본 콘텐츠 구조 보존
            original_content = {"text": content_data["text"]}
            if content_data["tables"]:
                original_content["tables"] = content_data["tables"]
            if content_data["images"]:
                original_content["images"] = content_data["images"]

            # 최소한의 유용한 메타데이터로 최종 검색 단위 조립
            processed_chunk = {
                "content": enhanced_content,
                "original_content": original_content,
                "type": content_data["types"],
                "page_number": get_page_number(chunk, i),
                "char_count": len(enhanced_content),
            }

            # processed_chunk 예시:
            # {
            #     "content": "AI-enhanced summary of the chunk... Image looks like this: <image_base64> ... Table looks like this: <table_html> ...",
            #     "original_content": {
            #         "text": "Full paragraph of the chunk...",
            #         "tables": ["<table><tr><th>Region</th><th>Revenue</th></tr><tr><td>APAC</td><td>$1.2M</td></tr></table>"],
            #         "images": ["iVBORw0KGgoAAA...base64..."]
            #     },
            #     "type": ["text", "table", "image"],
            #     "page_number": 3,
            #     "char_count": 142
            # }

            processed_chunks.append(processed_chunk)

        return processed_chunks
    except Exception as e:
        raise Exception(f"Failed to summarise chunks: {str(e)}")


def vectorize_chunks_summary_and_store_in_database(processed_chunks, document_id):
    """청크의 AI 요약에 대한 벡터 임베딩을 생성하여 데이터베이스에 저장합니다."""

    try:
        # Step 1 : 청크 벡터화
        ai_summary_list = [chunk["content"] for chunk in processed_chunks]
        # ai_summary_list = ["Ai-enhanced summary of the chunk...", "Ai-enhanced summary of the chunk...", ...]

        # 엣지 케이스: 청크가 많을수록 API 호출도 증가. API 한도 초과 방지를 위해 배치로 처리
        batch_size = 10
        all_vectorized_embeddings = []

        for start in range(0, len(ai_summary_list), batch_size):

            # batch_size(10) 단위로 분할
            end = start + batch_size
            batch_texts = ai_summary_list[start:end]  # 10개 이하의 청크 배치 추출

            # 지수 백오프(exponential backoff)를 사용한 간단한 재시도
            attempt = 0
            while True:
                try:
                    embeddings = openAI["embeddings"].embed_documents(batch_texts)
                    # As
                    all_vectorized_embeddings.extend(
                        embeddings
                    )  # 'extend' - 리스트 끝에 여러 요소를 추가하는 내장 list 메서드
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= 3:
                        raise e
                    time.sleep(2**attempt)

        # Step 2 : 임베딩과 함께 청크 저장
        # chunk_embedding_pairs: (processed_chunk, embedding_vector) 튜플의 목록
        # 예시:
        # [
        #     ({"content": "...", "page_number": 1, "type": ["text"]}, [0.123, -0.456, 0.789, ...]),
        #     ({"content": "...", "page_number": 2, "type": ["text", "table"]}, [0.234, -0.567, 0.890, ...]),
        #     ...
        # ]
        chunk_embedding_pairs = list(zip(processed_chunks, all_vectorized_embeddings))
        stored_chunk_ids = []

        for i, (processed_chunk, embedding_vector) in enumerate(chunk_embedding_pairs):
            # 각 processed_chunk에 document_id, chunk_index, embedding 추가
            # chunk_data_with_embedding 예시:
            # {
            #     * 위와 동일하나 document_id, chunk_index, embedding 추가
            #     "content": "AI-enhanced summary of the chunk...","original_content": {"text": "...", "tables": ["<table>...</table>"], "images": ["<base64>"]},"type": ["text", "table", "image"],"page_number": 3,"char_count": 142,
            #     "document_id": "doc_123",
            #     "chunk_index": 0,
            #     "embedding": [0.123, -0.456, 0.789, 0.234, ...]  # 1536 dimensions
            # }
            chunk_data_with_embedding = {
                **processed_chunk,
                "document_id": document_id,
                "chunk_index": i,
                "embedding": embedding_vector,
            }

            result = (
                supabase.table("document_chunks")
                .insert(chunk_data_with_embedding)
                .execute()
            )
            stored_chunk_ids.append(result.data[0]["id"])

        # print(f"Successfully stored {len(processed_chunks)} chunks with embeddings")
        return stored_chunk_ids

    except Exception as e:
        raise Exception(f"Failed to vectorize chunks and store in database: {str(e)}")