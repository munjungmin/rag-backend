# from unstructured.partition.html import partition_html
# from unstructured.partition.pdf import partition_pdf
# from unstructured.partition.docx import partition_docx
# from unstructured.partition.pptx import partition_pptx
# from unstructured.partition.text import partition_text
# from unstructured.partition.md import partition_md

from src.services.llm import openAI
from langchain_core.messages import HumanMessage


def partition_document(temp_file: str, file_type: str, source_type: str = "file"):
    """파일 타입과 소스 타입에 따라 문서 파티셔닝"""

    # 지연 import
    from unstructured.partition.html import partition_html
    from unstructured.partition.pdf import partition_pdf

    source = (source_type or "file").lower()
    if source == "url":
        return partition_html(
            filename=temp_file,
        )

    kind = (file_type or "").lower()
    dispatch = {
        "pdf": lambda: partition_pdf(
            filename=temp_file,
            strategy="hi_res",  # 가장 정확한 (but 느린) 추출 처리 방식
            infer_table_structure=True,  # 테이블을 뒤섞인 텍스트가 아닌 구조화된 HTML로 유지
            extract_image_block_types=["Image"],  # PDF에서 발견된 이미지 추출
            extract_image_block_to_payload=True,  # 이미지를 payload에 base64 문자열로 저장
        ),
        # "docx": lambda: partition_docx(
        #     filename=temp_file,
        #     strategy="hi_res",
        #     infer_table_structure=True,
        #     # ! docx, pptx, md 파일의 이미지 추출은 미구현 상태
        # ),
        # "pptx": lambda: partition_pptx(
        #     filename=temp_file,
        #     strategy="hi_res",
        #     infer_table_structure=True,
        # ),
        # "txt": lambda: partition_text(filename=temp_file),
        # "md": lambda: partition_md(filename=temp_file),
    }

    if kind not in dispatch:
        raise ValueError(f"Unsupported file_type: {file_type}")

    return dispatch[kind]()


def analyze_elements(elements):
    """elements를 분석하고 요약 반환"""

    text_count = 0
    table_count = 0
    image_count = 0
    title_count = 0
    other_count = 0

    # 각 element를 순회하며 유형 카운트
    for element in elements:
        element_name = type(
            element
        ).__name__  # __name__은 "Table"이나 "NarrativeText" 같은 클래스명을 반환하는 특수 속성

        if element_name == "Table":
            table_count += 1
        elif element_name == "Image":
            image_count += 1
        elif element_name in ["Title", "Header"]:
            title_count += 1
        elif element_name in ["NarrativeText", "Text", "ListItem", "FigureCaption"]:
            text_count += 1
        else:
            other_count += 1

    # 간단한 딕셔너리 반환
    return {
        "text": text_count,
        "tables": table_count,
        "images": image_count,
        "titles": title_count,
        "other": other_count,
    }


def separate_content_types(chunk, source_type="file"):
    """청크 내 콘텐츠 유형 분석"""
    is_url_source = source_type == "url"

    content_data = {
        "text": chunk.text,  # 모든 청크는 기본적으로 텍스트를 가지므로 chunk.text는 None이 아님
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    # 원본 elements에서 테이블과 이미지 확인
    if hasattr(chunk, "metadata") and hasattr(
        chunk.metadata, "orig_elements"
    ):  # orig_elements는 청크 내 모든 원자적 element를 나열함
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # 테이블 처리
            if element_type == "Table":
                content_data["types"].append("table")
                # getattr은 객체의 지정된 속성 값을 반환하는 내장 함수입니다.
                # text_as_html이 존재하면 테이블의 HTML 표현을 반환하고, 없으면 element의 text 속성을 반환합니다.
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            # 이미지 처리 (URL 소스는 건너뜀)
            elif element_type == "Image" and not is_url_source:
                if (
                    hasattr(element, "metadata")
                    and hasattr(element.metadata, "image_base64")
                    and element.metadata.image_base64 is not None
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))

    # https://www.youtube.com/watch?v=-vJ2-0RXkmk
    # Example return structure:
    # {
    #     "text": "This is the main text content of the chunk...",
    #     "tables": ["<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>"],
    #     "images": ["iVBORw0KGgoAAAANSUhEUgAA..."],  # base64 encoded image strings
    #     "types": ["text", "table", "image"]  # or ["text"], ["text", "table"], etc.
    # }

    return content_data


def get_page_number(chunk, chunk_index):
    """청크에서 페이지 번호를 가져오거나 대체값 사용"""
    if hasattr(chunk, "metadata"):
        page_number = getattr(chunk.metadata, "page_number", None)
        if page_number is not None:
            return page_number

    # 대체: 청크 인덱스를 페이지 번호로 사용
    return chunk_index + 1


def create_ai_summary(text, tables_html, images_base64):
    """청크에 포함된 테이블과 이미지에 대한 AI 강화 요약 생성"""

    try:
        # 더 효율적인 지시사항으로 텍스트 prompt 구성
        prompt_text = f"""
            Create a searchable index for this document content.
            CONTENT:
            {text}
        """

        # 테이블이 있는 경우 추가
        if tables_html:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables_html):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        # 더 간결하지만 효과적인 prompt
        prompt_text += """
            Generate a structured search index (aim for 250-400 words):

            QUESTIONS: List 5-7 key questions this content answers (use what/how/why/when/who variations)

            KEYWORDS: Include:
            - Specific data (numbers, dates, percentages, amounts)
            - Core concepts and themes
            - Technical terms and casual alternatives
            - Industry terminology

            VISUALS (if images present):
            - Chart/graph types and what they show
            - Trends and patterns visible
            - Key insights from visualizations

            DATA RELATIONSHIPS (if tables present):
            - Column headers and their meaning
            - Key metrics and relationships
            - Notable values or patterns

            Focus on terms users would actually search for. Be specific and comprehensive.

            SEARCH INDEX:"""

        # 텍스트 prompt로 시작하는 메시지 콘텐츠 구성
        message_content = [{"type": "text", "text": prompt_text}]

        # 메시지에 이미지 추가
        for i, image_base64 in enumerate(images_base64):
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )
            # print(f"🖼️ Image {i+1} included in summary request")

        message = HumanMessage(content=message_content)
        response = openAI["chat_llm"].invoke([message])

        return response.content

    except Exception as e:
        raise Exception(f"Failed to create AI summary: {str(e)}")