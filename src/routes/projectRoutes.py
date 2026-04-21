from fastapi import APIRouter, Depends, HTTPException
from src.rag.retrieval.retrieval import retrieve_context
from src.rag.retrieval.utils import prepare_prompt_and_invoke_llm
from src.services.supabase import supabase
from src.models.schemas import MessageCreate, MessageRole, ProjectCreate, ProjectSettings
from src.services.clerkAuth import get_current_user_clerk_id

router = APIRouter(tags=["projects"])

"""
`/api/projects`

GET    /api/projects                          - Get all projects for authenticated user
POST   /api/projects                          - Create a new project with default settings
GET    /api/projects/{project_id}             - Get a specific project by ID
DELETE /api/projects/{project_id}             - Delete a project and all related data

GET    /api/projects/{project_id}/chats       - Get all chats for a project

GET    /api/projects/{project_id}/settings    - Get settings for a project
PUT    /api/projects/{project_id}/settings    - Update settings for a project
POST   /api/projects/{project_id}/chats/{chat_id}/messages   -Send a message to a Specific Chat
"""


@router.get("/")
def get_projects(clerk_id: str = Depends(get_current_user_clerk_id)):
    try:
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()
        
        return {
            "success": True,
            "message": "Projects retrieved successfully",
            "data": result.data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")


@router.post("/")
def create_project(project: ProjectCreate, clerk_id: str = Depends(get_current_user_clerk_id)):
    try:
        # 데이터베이스에 새 프로젝트 삽입
        project_result = supabase.table("projects").insert({
            "name": project.name,
            "description": project.description,
            "clerk_id": clerk_id
        }).execute()

        if not project_result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create project"
            )
        
        created_project = project_result.data[0]
        project_id = created_project["id"]

        # 프로젝트 기본 설정 생성
        settings_result = supabase.table("project_settings").insert({
            "project_id": project_id,
            "embedding_model": "text-embedding-3-large",
            "rag_strategy": "basic",
            "agent_type":"agentic",
            "chunks_per_search":10,
            "final_context_size": 5,
            "similarity_threshold": 0.3,
            "number_of_queries": 5,
            "reranking_enabled": True,
            "reranking_model": "rerank-english-v3.0",
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }).execute()

        if not settings_result.data:
            # 롤백 - 설정 생성 실패 시 프로젝트 삭제
            supabase.table("projects").delete().eq("id", project_id).execute()
            raise HTTPException(
                status_code=500,
                detail="Failed to create project settings"
            )
        
        return {
            "success": True,
            "message": "Project created successfully",
            "data": created_project
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create project: {str(e)}"
        )


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        # 프로젝트가 존재하고 사용자 소유인지 확인
        project_result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or access denied"
            )

        # 프로젝트 삭제 (CASCADE로 관련 데이터 자동 삭제): project settings, documents, chunks가 자동으로 삭제됨
        deleted_result = supabase.table("projects").delete().eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not deleted_result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete project"
            )
        
        return {
            "success": True,
            "message": "Project deleted successfully",
            "data": deleted_result.data[0]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete project: {str(e)}"
        )


@router.get("/{project_id}")
def get_project(
    project_id: str,
    clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        project_result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found"
            )
    
        return {
            "success": True,
            "message": "Project retrieved successfully",
            "data": project_result.data[0]
        }
    
    except HTTPException:
        raise    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get project: {str(e)}"
        )
    

@router.get("/{project_id}/chats")
def get_project_chats(
    project_id: str,
    clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        result = supabase.table("chats").select("*").eq("project_id", project_id).eq("clerk_id", clerk_id).order("created_at", desc=True).execute()

        return {
            "success": True,
            "message": "Project chats retrieved successfully",
            "data": result.data     # supabase는 결과가 없어도 null이 아니라 [] 반환
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project chats: {str(e)}"
        )


@router.get("/{project_id}/settings")
def get_project_settings(
    project_id: str,
    clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        result = supabase.table("project_settings").select("*").eq("project_id", project_id).execute() 

        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Project settings not found" 
            )
        
        return {
            "success": True,
            "message": "Project settings retrieved successfully",
            "data": result.data[0]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while updating project settings: {str(e)}"
        )

@router.put("/{project_id}/settings")
async def update_project_settings(
    project_id: str,
    settings: ProjectSettings,
    clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        # 먼저 프로젝트가 존재하고 사용자 소유인지 확인
        project_result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data:
            raise HTTPException(
                status_code=404,
                detail=f"Project not found or access denied"
            )

        # 업데이트 실행
        result = supabase.table("project_settings").update(settings.model_dump()).eq("project_id", project_id).execute() # project_settings 테이블은 clerk_id 컬럼이 없으니 위에서 소유자 확인

        if not result.data:
            raise HTTPException( 
                status_code=404,
                detail=f"Project settings not found"
            )
        
        return {
            "success": True,
            "message": "Project settings updated successfully",
            "data": result.data[0]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update project settings: {str(e)}"
        )


@router.post("/{project_id}/chats/{chat_id}/messages")
async def send_message(
    project_id: str,
    chat_id: str,
    message: MessageCreate,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! 로직 흐름:
    * 1. 현재 사용자 clerk_id 조회
    * 2. 메시지를 데이터베이스에 삽입
    * 3. Retrieval (검색)
    * 4. Generation (검색된 컨텍스트 + 사용자 메시지)
    * 5. AI 응답을 데이터베이스에 삽입
    """
    try:
        # Step 1 : 메시지를 데이터베이스에 삽입
        message = message.content
        message_insert_data = {
            "content": message,
            "chat_id": chat_id,
            "clerk_id": current_user_clerk_id,
            "role": MessageRole.USER.value,
        }
        message_creation_result = (
            supabase.table("messages").insert(message_insert_data).execute()
        )

        if not message_creation_result.data:
            raise HTTPException(status_code=422, detail="Failed to create message")

        # Step 3 : 검색 (Retrieval)
        texts, images, tables, citations = retrieve_context(project_id, message)

        # Step 4 : 생성 (검색된 컨텍스트 + 사용자 메시지)
        final_response = prepare_prompt_and_invoke_llm(
            user_query=message, texts=texts, images=images, tables=tables
        )

        # Step 5: AI 응답을 데이터베이스에 삽입
        ai_response_insert_data = {
            "content": final_response,
            "chat_id": chat_id,
            "clerk_id": current_user_clerk_id,
            "role": MessageRole.ASSISTANT.value,
            "citations": citations,
        }
        ai_response_creation_result = (
            supabase.table("messages").insert(ai_response_insert_data).execute()
        )
        if not ai_response_creation_result.data:
            raise HTTPException(status_code=422, detail="Failed to create AI response")

        return {
            "message": "Message created successfully",
            "data": {
                "userMessage": message_creation_result.data[0],
                "aiMessage": ai_response_creation_result.data[0],
            },
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while creating message: {str(e)}",
        )


# clerk_id를 클라이언트가 서버에게 바로 넘기면, 해커가 그걸 탈취해서 delete projects api를 호출할 수도 있다. 
# 그래서 현대 앱들은 JWT 인증을 사용한다 
# Clerk도 내부적으로는 JWT 인증을 사용한다. 그래서 Clerk은 유저 회원가입 시, JWT 토큰을 주고, 브라우저는 이를 쿠키나 localStorage에 저장한다.  
# JWT 토큰은 암호화되어 있는 토큰이다. 그리고 그 안에 clerk_id를 숨겨놓았다.. 그래서 서버에서는 이를 decode할 수 있다 
# 클라이언트는 clerk_id를 바로 body에 담아서 전달하는게 아니라, authentication header에 JWT 토큰을 보내면 된다.  

# API 호출은 전부 header와 payload(body)를 가진다. header에는 token이 첨부되고, 서버는 그 토큰이 올바른지 검증한다 
# 이 API endpoint가 호출될 때마다, 검증

# 우리는 이걸 fast API method Depends() 를  사용해 할 수 있다.
# API가 호출될때, 특정 함수(clerk_id를 추출해서, 우리에게 전달해주는)를 실행시키고 싶을때  