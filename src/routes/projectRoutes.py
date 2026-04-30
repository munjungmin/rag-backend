from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from src.agents.simple_agent import create_rag_agent
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

def get_chat_history(chat_id: str) -> List[Dict[str, str]]:
    """
        최근 10개 채팅 내역을 가져오기
    """
    try:
        messages_result =(
            supabase.table("messages")
            .select("id, role, content")
            .eq("chat_id", chat_id) 
            .order("created_at", desc=True) # 최신순
            .limit(10)  
            .execute()
        )

        if not messages_result.data:
            return []
        
        recent_messages = list(reversed(messages_result.data)) # 시간순으로 다시 정렬

        formatted_history = []
        for msg in recent_messages:
            formatted_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        return formatted_history

    except Exception:
        # 만약 내역을 가져오지 못한다면 빈 리스트 리턴
        return []



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
        # Step 1: 새로운 채팅을 저장하기 전에 기존 채팅 내역 불러오기 
        chat_history = get_chat_history(chat_id)

        # Step 2 : 새 메시지 DB에 저장
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

        # Step 3 : agent 호출 
        agent = create_rag_agent(
            project_id=project_id,
            model="gpt-4o",
            chat_history=chat_history
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })

        final_response = result["messages"][-1].content
        citations = result.get("citations", [])

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
