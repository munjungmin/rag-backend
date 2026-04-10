from typing import Any, Dict, List, Tuple
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import supabase
from routers.auth import get_current_user
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize LLM for summarisation
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize embedding model
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536
)

router = APIRouter(
    tags=["chats"]
)

class ChatCreate(BaseModel):
    title: str
    project_id: str


@router.post("/api/chats")
async def create_chat(
    chat: ChatCreate,
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("chats").insert({
            "title": chat.title,
            "project_id": chat.project_id,
            "clerk_id": clerk_id
        }).execute()

        return {
            "success": True,
            "message": "Chat created successfully",
            "data": result.data[0]
        }        

    except Exception as e: 
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create new chat: {str(e)}"
        )


@router.delete("/api/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("chats").delete().eq("id", chat_id).eq("clerk_id", clerk_id).execute()

        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Chat not found or access denied"
            )
        
        return {
            "success": True,
            "message": "Chat deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete chat: {str(e)}"
        )

@router.get("/api/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Get the chat and verify it belongs to the user AND has a project_id
        result = supabase.table("chats").select("*").eq("id", chat_id).eq("clerk_id", clerk_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Chat not found or access denied")
        
        chat = result.data[0]

        # Get messages for this chat
        messages_result = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at", desc=False).execute()

        # Add messages to chat object
        chat['messages'] = messages_result.data or []

        return {
            "message": "Chat retrieved successfully",
            "data": chat
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve chat: {str(e)}"
        )

def load_project_settings(project_id: str) -> dict:
    """ Load project settings from database """
    print(f"⚙️ Fetching project settings...")
    settings_result = supabase.table("project_settings").select("*").eq("project_id", project_id).execute()

    if not settings_result.data:
        raise HTTPException(status_code=404, detail="Project settings not found")

    settings = settings_result.data[0]
    print(f"✅ Settings retrieved")

    return settings
    

def get_document_ids(project_id: str) -> List[str]:
    """ Get all documents IDs for a project """
    print(f"📄 Fetching project documents...")
    documents_result = supabase.table("project_documents").select("id").eq("project_id", project_id).execute()
    
    document_ids = [doc['id'] for doc in documents_result.data]
    print(f"✅ Found {len(document_ids)} documents")

    return document_ids


def vector_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    """ Execute vector search """
    query_embedding = embeddings_model.embed_query(query)
    
    result = supabase.rpc("vector_search_document_chunks", {
        "query_embedding": query_embedding,
        "filter_document_ids": document_ids,
        "match_threshold": settings["similarity_threshold"],
        "chunks_per_search": settings['chunks_per_search']
    }).execute()

    return result.data if result.data else []

def keyword_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    """ Execute keyword search"""
    result = supabase.rpc("keyword_search_document_chunks", {
        "query_text": query,
        "filter_document_ids": document_ids,
        "chunks_per_search": settings["chunks_per_search"]
    }).execute()

    return result.data if result.data else []

def rrf_rank_and_fuse(search_results_list: list[list[dict]], weights: list[float] = None, k: int = 60) -> list[dict]:
    """ RRF (Reciprocal Rank Fusion) ranking """
    if not search_results_list or not any(search_results_list): # 리스트가 비어있거나, 리스트 안의 모든 원소가 비어있으면 
        return []
    
    if weights is None:
        weights = [1.0 / len(search_results_list)] * len(search_results_list)

    chunk_scores = {}
    all_chunks = {}

    for search_idx, results in enumerate[list[dict]](search_results_list):
        weight = weights[search_idx]

        for rank, chunk in enumerate[dict](results):
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue

            rrf_score = weight * (1.0 / (k + rank + 1))
            
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] += rrf_score
            else:
                chunk_scores[chunk_id] = rrf_score
                all_chunks[chunk_id] = chunk

    sorted_chunk_ids = sorted(chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True)
    return [all_chunks[chunk_id] for chunk_id in sorted_chunk_ids]


def hybrid_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    """ Execute hybrid search by combining vector and keyword results"""

    # Get results from both search methods
    vector_results = vector_search(query, document_ids, settings)
    keyword_results = keyword_search(query, document_ids, settings)

    print(f"📈 Vector search returned: {len(vector_results)} chunks")
    print(f"📈 Keyword search returned: {len(keyword_results)} chunks")

    # Combine using RRF with configured weights
    return rrf_rank_and_fuse(
        [vector_results, keyword_results],
        [settings["vector_weight"], settings["keyword_weight"]]
    )


class QueryVariations(BaseModel):
    queries: List[str]

def  generate_query_variations(original_query: str, num_queries: int = 3) -> list[str]:
    """ Generate query variations using LLM """
    system_prompt = f"""Generate {num_queries-1} alternative ways to phrase this question for document search. Use different keywords and synonyms while maintaining the same intent. Return exactly {num_queries-1} variations."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original query: {original_query}")
        ]

        structured_llm = llm.with_structured_output(QueryVariations)
        result = structured_llm.invoke(messages)

        return [original_query] + result.queries[:num_queries-1]
    except Exception:
        return [original_query]

def build_context(chunks: List[Dict]) -> Tuple[List[str], List[str], List[str], List[Dict]]:
    """
    Returns:
        Tuple of (texts, images, tables, citations)
    """
    if not chunks: 
        return [], [], [], []
    
    texts = []
    images = []
    tables = []
    citations = []

    # Batch fetch all filenames in ONE query
    doc_ids = [chunk['document_id'] for chunk in chunks if chunk.get('document_id')]
    unique_doc_ids: List[str] = list[str](set[str](doc_ids)) 

    filename_map = {}

    if unique_doc_ids:
        result = supabase.table("project_documents")\
            .select("id, filename")\
            .in_("id", unique_doc_ids)\
            .execute()
        filename_map = {doc['id']: doc['filename'] for doc in result.data}

    # Process each chunk
    for chunk in chunks:
        original_content = chunk.get('original_content', {})

        # Extract content from chunk
        chunk_text = original_content.get('text', '')
        chunk_images = original_content.get('images', [])
        chunk_tables = original_content.get('tables', [])

        # Collect content
        if chunk_text:
            texts.append(chunk_text) 
        images.extend(chunk_images) # image와 table은 list 형태라서 extend, text는 str이라 append
        tables.extend(chunk_tables)

        # Add citation for every chunk
        doc_id = chunk.get('document_id')
        if doc_id:
            citations.append({
                "chunk_id": chunk.get('id'),
                "document_id": doc_id,
                "filename": filename_map.get(doc_id, "Unknown document"),
                "page": chunk.get('page_number', "Unknown")
            })
    
    return texts, images, tables, citations

def validate_context(texts: List[str], images: List[str], tables: List[str], citations: List[Dict]) -> None:
    """Validate and print context data in a readable format"""
    print("\n" + "="*80)
    print("📦 CONTEXT VALIDATION")
    print("="*80)
    
    # Texts - SHOW FULL TEXT
    print(f"\n📝 TEXTS: {len(texts)} chunks")
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*80}")
        print(f"CHUNK [{i}] - {len(text)} characters")
        print(f"{'='*80}")
        print(text)  # ✅ Full text, no truncation
        print(f"{'='*80}\n")
    
    # Images
    print(f"\n🖼️  IMAGES: {len(images)}")
    for i, img in enumerate(images, 1):
        img_preview = str(img)[:60] + ('...' if len(str(img)) > 60 else '')
        print(f"  [{i}] {img_preview}")
    
    # Tables
    print(f"\n📊 TABLES: {len(tables)}")
    for i, table in enumerate(tables, 1):
        if isinstance(table, dict):
            rows = len(table.get('rows', []))
            cols = len(table.get('headers', []))
            print(f"  [{i}] {rows} rows × {cols} cols")
        else:
            print(f"  [{i}] Type: {type(table).__name__}")
    
    # Citations
    print(f"\n📚 CITATIONS: {len(citations)}")
    for i, cite in enumerate(citations, 1):
        chunk_id = cite['chunk_id'][:8] if cite.get('chunk_id') else 'N/A'
        print(f"  [{i}] {cite['filename']} (pg.{cite['page']}) | chunk: {chunk_id}...")
    
    # Summary
    total_chars = sum(len(text) for text in texts)
    print(f"\n{'='*80}")
    print(f"✅ Total: {len(texts)} texts ({total_chars:,} chars), {len(images)} images, {len(tables)} tables, {len(citations)} citations")
    print("="*80 + "\n")

def prepare_prompt_and_invoke_llm(
    user_query: str,
    texts: List[str],
    images: List[str],
    tables: List[str]
) -> str:
    """
        Builds system prompt with context and invokes LLM with multi-modal support

        Args:
            user_query: The user's question
            texts: List of text chunks from documents
            images: List of base64-encoded images
            tables: List of HTML table strings

        Returns:
            AI response string
    """

    # Build system prompt parts
    prompt_parts = []

    # Main instruction
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
    
    # Add text contexts
    if texts:
        prompt_parts.append("=" * 80)
        prompt_parts.append("CONTEXT DOCUMENTS")
        prompt_parts.append("=" * 80 + "\n")
        
        for i, text in enumerate(texts, 1):
            prompt_parts.append(f"--- Document Chunk {i} ---")
            prompt_parts.append(text.strip())
            prompt_parts.append("")
    
    # Add tables if present
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
    
    # Reference images if present
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
    
    # Final instruction
    prompt_parts.append("=" * 80)
    prompt_parts.append(
        "Based on all the context provided above (documents, tables, and images), "
        "please answer the user's question accurately and comprehensively."
    )
    prompt_parts.append("=" * 80)

    system_prompt = "\n".join(prompt_parts)

    # Build messages for LLM
    messages = [SystemMessage(content=system_prompt)]

    # Create human message with user query and images
    if images:
        # Multi-modal message: text + images
        content_parts = [{"type": "text", "text": user_query}]

        # Add each image to the content array
        for img_base64 in images:
            # Clean base64 string if it has data URI preifx
            if img_base64.startswith('data:image'):
                img_base64 = img_base64.split(",", 1)[1]
            
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64, {img_base64}"}
            })

        messages.append(HumanMessage(content=content_parts))
    else:
        # Text-only message
        messages.append(HumanMessage(content=user_query))
    
    # Invoke LLM and return response
    print(f"🤖 Invoking LLM with {len(messages)} messages ({len(texts)} texts, {len(tables)} tables, {len(images)} images) ")
    response = llm.invoke(messages)

    return response.content



class SendMessageRequest(BaseModel):
    content: str

@router.post("/api/projects/{project_id}/chats/{chat_id}/messages")
async def send_message(
    chat_id: str,
    project_id: str,
    request: SendMessageRequest,
    clerk_id: str = Depends(get_current_user)
):
    """
        User message -> LLM -> AI response
    """
    try:
        message = request.content

        print(f"💬 New message: {message[:50]}...")

        # 1. Save user message
        print(f"💾 Saving user message...")
        user_message_result = supabase.table("messages").insert({
            "chat_id": chat_id,
            "content": message,
            "role": "user",
            "clerk_id": clerk_id
        }).execute()

        user_message = user_message_result.data[0]
        print(f"✅ User message saved: {user_message['id']}")

        # 2. Load project settings
        # to know: chunk size, similarity threshold, etc.
        settings = load_project_settings(project_id)

        # 3. Get document IDs for this project
        # to narrow search scope to only documents uploaded to this specific project 
        document_ids = get_document_ids(project_id)

        strategy = settings["rag_strategy"]
        print(f"\n 🔎 RAG Strategy: {strategy.upper()}")

        # 4. Generate query embedding
        # 5. Perform vector search using the RPC function
        # return only chunks above a similarity threshold, sorted by relevance, limited to the top N results.

        if strategy == "basic": # vector search
            chunks = vector_search(message, document_ids, settings)
            print(f"✅ Retrieved {len(chunks)} relevant chunks from vector search")

        elif strategy == "hybrid":
            print(" Executing: Hybrid Search (Vector + Keyword)")
            chunks = hybrid_search(message, document_ids, settings)
            print(f" Hybrid search returned: {len(chunks)} chunks")

        elif strategy == "multi-query-vector":
            print(f" Executing: Multi-Query Vector Search ({settings["number_of_queries"]} queries)")
            queries = generate_query_variations(message, settings["number_of_queries"])
            print(f"Generated queries: {queries}")

            all_results = []
            for i, q in enumerate[str](queries):
                results = vector_search(q, document_ids, settings)
                print(f"Query {i+1} '{q}' returned: {len(results)} chunks")
                all_results.append(results)
            chunks = rrf_rank_and_fuse(all_results)
            print(f" RRF fusion returned: {len(chunks)} chunks")
             
        elif strategy == "multi-query-hybrid":
            print(f" Executing: Multi-Query Hybrid Search ({settings["number_of_queries"]} queries)")
            queries = generate_query_variations(message, settings["number_of_queries"])
            print(f"Generated queries: {queries}")

            # Stage 1: Per-query hybrid fusion
            all_hybrid_results = []
            for i, q in enumerate[str](queries):
                print(f"\n Query {i+1}: '{q}")

                # Use the existing hybrid_search function which handles weights
                hybrid_results = hybrid_search(q, document_ids, settings)

                print(f"Hybrid fusion returned: {len(hybrid_results)} chunks")

                all_hybrid_results.append(hybrid_results)

            # Stage 2: Cross-query fusion
            print(f"\nFinal RRF fusion across {len(all_hybrid_results)} queries")
            chunks = rrf_rank_and_fuse(all_hybrid_results)
            print(f" Final result: {len(chunks)} chunks")

        # Trim to final context size
        chunks = chunks[:settings["final_context_size"]]

        # 6. Build context from retrieved chunks
        # Format the retrieved chunks into a structured context with citations
        texts, images, tables, citations = build_context(chunks)
        validate_context(texts, images, tables, citations)


        # 7. Build system prompt with injected context 
        # Add the retrieved document context to the system prompt so the LLM can answer based on the document
        print(f"🤖 Preparing context and calling LLM...")
        ai_response = prepare_prompt_and_invoke_llm(
            user_query=message,
            texts=texts,
            images=images,
            tables=tables
        )
        print(f"✅ LLM response received ({len(ai_response)}) characters")

        print(f"💾 Saving AI message...")
        ai_message_result = supabase.table("messages").insert({
            "chat_id": chat_id,
            "content": ai_response,
            "role": "assistant",
            "clerk_id": clerk_id,
            "citations": citations
        }).execute()

        ai_message = ai_message_result.data[0]
        print(f"✅ AI message saved: {ai_message['id']}")

        # Return data
        return {
            "message": "Messages send successfully",
            "data": {
                "userMessage": user_message,
                "aiMessage": ai_message
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve chat: {str(e)}"
        )
