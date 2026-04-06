import os
from celery import Celery
from database import supabase, s3_client, BUCKET_NAME
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage



REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# Initialize LLM for summarisation
llm = ChatOpenAI(model="gpt-4o", temperature=0)

 
# Initialize embedding model
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536
)

# Create Celery app
celery_app = Celery(
    'document_processor', # Name of our Celery app,
    broker=REDIS_URL, # where tasks are queued
    backend=REDIS_URL # task 결과가 저장될 곳
)

def update_status(document_id: str, status: str, details: dict = None):
    """Update document process status with optional details"""

    # Get current document 
    result = supabase.table("project_documents").select("processing_details").eq("id", document_id).execute()

    # Start with existing details or empty dict
    current_details = {}

    if result.data and result.data[0]["processing_details"]:
        current_details = result.data[0]["processing_details"]

    # Add new details if provided
    if details: 
        current_details.update(details)

    # Update document
    supabase.table("project_documents").update({
        "processing_status": status,
        "processing_details": current_details
    }).eq("id", document_id).execute()

@celery_app.task
def process_document(document_id: str):
    """
        Real document Processing
    """

    try:

        doc_result = supabase.table("project_documents").select("*").eq("id", document_id).execute()
        document = doc_result.data[0]
        source_type = document.get('source_type', 'file')
        # Step 1: Download and partition
        update_status(document_id, "partitioning")
        elements = download_and_partition(document_id, document)

        tables = sum(1 for e in elements if e.category == "Table")
        images = sum(1 for e in elements if e.category == "Image")
        text_elements = sum(1 for e in elements if e.category in ["NarrativeText", "Title", "Text"])
        print(f"📊Extracted: {tables} tables, {images} images, {text_elements} text elements") 


        # Step 2: Chunk elements  
        chunks, chunking_metrics = chunk_elements_by_title(elements) # 메모리에서 문자열을 자르는 작업이라 매우 빠르고, synchronous한 작업
        update_status(document_id, "summarising", {
            "chunking": chunking_metrics
        })


        # Step 3: Summarising chunks 
        processed_chunks = summarise_chunks(chunks, document_id, source_type)

        # Step 4: Vectorization & storing 
        update_status(document_id, "vectorization")
        stored_chunk_ids = store_chunks_with_embeddings(document_id, processed_chunks)

        # Mark as completed
        update_status(document_id, 'completed')
        print(f"✅ Celery task completed for document: {document_id} with {len(stored_chunk_ids)} chunks")

        # 결과는 어차피 redis에 저장됨 -> 어차피 사용안해서 삭제해도 됨
        return {
            "status" : "success",
            "document_id" : document_id
        }
    
    except Exception as e:
        print(f"Error: {e}")
        raise


def download_and_partition(document_id: str, document: dict):
    """ Download document from S3 / Crawl URL and partition into elements """

    print(f"Downloading and partitioning document {document_id}")

    source_type = document.get("source_type", "file")

    if source_type == "url":
        # Crawl URL

        pass
    else:
        # Handle file processing
        s3_key = document["s3_key"]
        filename = document["filename"]
        file_type = filename.split(".")[-1].lower()

        # Download to a temporary location, 파티셔닝이 끝나면 삭제 
        temp_file = f"/tmp/{document_id}.{file_type}"
        s3_client.download_file(BUCKET_NAME, s3_key, temp_file) # synchrocous - 다운로드가 끝나면 다음 줄로 넘어감 (celery가 별도 프로세스/스레드에서 실행하니까 괜찮)

        elements = partition_document(temp_file, file_type, source_type="file")

    elements_summary = analyze_elements(elements)

    update_status(document_id, "chunking", {
        "partitioning": {
            "elements_found": elements_summary
        }
    })
    os.remove(temp_file)

    return elements


def partition_document(temp_file: str, file_type: str, source_type: str = "file"):
    """Partition document based on file type and source type"""

    from unstructured.partition.pdf import partition_pdf # celery-worker container에서 실행

    if source_type == "url":
        pass 

    if file_type == "pdf":
        return partition_pdf(
            filename=temp_file,  # Path to PDF file
            strategy="hi_res",      # Use the most accurate (but slower) processing method of extraction
            infer_table_structure=True,     # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"],    # Grab images found in the PDF
            extract_image_block_to_payload=True     # Store images as base64 data you can actually use
        )
    
def analyze_elements(elements):
    """ Count different types of elements found in the document"""

    text_count = 0
    table_count = 0
    image_count = 0
    title_count = 0
    other_count = 0

    # Go through each element and count what type it is
    for element in elements:
        element_name = type(element).__name__ # Get the class name like "Table" or "NarrativeText"
        
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

    # Return a simple dictionary
    return {
        "text": text_count,
        "tables": table_count,
        "images": image_count,
        "titles": title_count,
        "other": other_count
    }


def chunk_elements_by_title(elements):
    """ Chunk elements using title-based strategy and collect metrics """
    
    from unstructured.chunking.title import chunk_by_title

    chunks = chunk_by_title(
        elements, # partition된 PDF 요소들
        max_characters=3000, # Hard limit - 청크는 3000자를 절대 넘지 않음  
        new_after_n_chars=2400, # 2400자가 넘으면 새로운 청크로 생성 시도 
        combine_text_under_n_chars=500 # 500자 이하의 청크들은 주변끼리 합치기  
    )

    # Collect chunking metrics
    total_chunks = len(chunks)

    # 청크 통계/지표, 나중에 평균 청크 크기 등을 추가하면 됨 
    chunking_metrics = {        
        "total_chunks" : total_chunks
    }

    print(f"✅ Created {total_chunks} chunks from {len(elements)} elements")

    return chunks, chunking_metrics


def summarise_chunks(chunks, document_id, source_type="file"):
    """Transform chunks into searchable content with AI summaries (image, table에 메타데이터 추가)"""
    print("🧠 Processing chunks with AI Summarisation...")

    processed_chunks = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1

        # Update progress directly : 청크 요약은 청크마다 대략 5-8초 정도이므로 유저에게 진행률을 보여주기 위해 update  
        update_status(document_id, "summarising", {
            "summarising": {
                "current_chunk": current_chunk,
                "total_chunks": total_chunks
            }
        })

        # Extract content from the chunk
        content_data = seperate_content_types(chunk, source_type)

        # Debug prints
        print(f"    Types found: {content_data['types']}")
        print(f"    Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        # Decide if we need AI summarisation
        if content_data['tables'] or content_data['images']:
            print(f"    Creating AI summary for mixed content...")
            enhanced_content = create_ai_summary(
                content_data['text'],
                content_data['tables'],
                content_data['images'],
            )
        else:
            enhanced_content = content_data['text']  
        
        # Build the original_content structure
        original_content = {'text': content_data['text']}
        if content_data['tables']:
            original_content['tables'] = content_data["tables"]
        if content_data["images"]:
            original_content["images"] = content_data["images"]

        # Create processed chunk with all data
        processed_chunk = {
            'content': enhanced_content,
            'original_content': original_content,
            'type': content_data['types'],
            'page_number': get_page_number(chunk, i),
            'char_count': len(enhanced_content)
        }

        processed_chunks.append(processed_chunk)

    print(f"✅ Processed {len(processed_chunks)} chunks")
    return processed_chunks


def seperate_content_types(chunk, source_type="file"):
    """Analyze what types of content are in a chunk"""
    is_url_source = source_type == "url"

    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle tables
            if element_type == 'Table':        # table은 URL이든 PDF든 HTML text로 추출되기 때문에 source type에 관계없이 동일하게 처리 
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text) 
                content_data['tables'].append(table_html)

            # Handle images (skip for URL sources - URL이면 <img src ="..."> 으로 참조될 뿐, 바이너리 데이터를 직접 가져오지 않아서 스킵)
            if element_type == 'Image' and not is_url_source:
                if (hasattr(element.metadata, 'image_base64') and
                    element.metadata.image_base64 is not None):

                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)

    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_summary(text, tables_html, images_base64):
    """Create AI-enhanced summary for mixed content"""

    try:
        # Build the text prompt with more efficient instructions
        prompt_text = f"""Create a searchable index for this documnet content.

CONTENT:
{text}

"""
        # Add tables if present
        if tables_html:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables_html):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        # More concise but effective prompt (벡터 검색에 최적화된 텍스트를 만들라는 프롬프트)
        # Questions: 이 문서가 답할 수 있는 질문을 미리 만들어라 
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

SEARCH INDEX:
"""

        # Build mesage content starting with the text prompt
        message_content = [{"type": "text", "text": prompt_text}]

        # Add images to the message
        for i, images_base64 in enumerate(images_base64):
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{images_base64}"}     # Open AI가 요구하는 data URL 형식으로 만들기
            })
            print(f"🖼️ Image {i+1} included in summary request")

        message = HumanMessage(content=message_content)

        response = llm.invoke([message])

        return response.content
    
    except Exception as e:
        print(f" AI summary failed: {e}")


def get_page_number(chunk, chunk_index):
    """Get page number from chunk or use fallback"""
    if hasattr(chunk, 'metadata'):
        page_number = getattr(chunk.metadata, 'page_number', None)
        if page_number is not None:
            return page_number
        
    # Fallback: use chunk index as page number
    return chunk_index + 1


def store_chunks_with_embeddings(documnet_id: str, processed_chunks: list):
    """Generate embeddings and store chunks in one efficient operation"""
    print("Generating embeddings and storing chunks...")

    if not processed_chunks:
        print("No chunks to process")
        return []
    
    # Step 1: Generate embeddings for all chunks
    print(f"Generating embeddings for {len(processed_chunks)} chunks...")

    # Extract content for embedding generation
    texts = [chunk_data['content'] for chunk_data in processed_chunks] # 청크 데이터가 너무 많으면 한번의 api 호출로 전부 전달할 수 없음 -> 배치 사용 

    # Generate embeddings in batches to avoid API limits
    batch_size = 10
    all_embeddings = []

    for i in range(0, len(texts), batch_size): # batch_size 간격으로 
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        print(f"✅ Generated embeddings for batch {i/batch_size + 1}/{(len(texts) + batch_size -1)//batch_size}")

    # Step 2: Store chunks with embeddings
    print("Storing chunks with embeddings in database...")         
    stored_chunk_ids = []

    for i, (chunk_data, embedding) in enumerate(zip(processed_chunks, all_embeddings)): # zip 두 리스트를 쌍으로 묶어줌 [a,b][1,2] -> [(a,1), (b,2)]
        # Add document_id, chunk_index, and embedding
        chunk_data_with_embedding = {
            **chunk_data,
            'document_id': documnet_id,
            'chunk_index': i,
            'embedding': embedding
        }

        result = supabase.table('document_chunks').insert(chunk_data_with_embedding).execute()
        stored_chunk_ids.append(result.data[0]['id'])

    print(f"Successfully stored {len(processed_chunks)} chunks with embeddings")
    return stored_chunk_ids

