from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import supabase
from routers.auth import get_current_user

router = APIRouter(
    tags=["projects"]
)

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


@router.get("/api/projects")
def get_projects(clerk_id: str = Depends(get_current_user)):
    """
    Retrieve all projects for the authenticated user
    """
      
    try:
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()
        
        return {
            "success": True,
            "message": "Projects retrieved successfully",
            "data": result.data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")


@router.post("/api/projects")
def create_project(project: ProjectCreate, clerk_id: str = Depends(get_current_user)):  # pydantic model: 프론트가 보낸 데이터를 ProjectCreate에 넣고 검증해줌 
    """
    Create a new project with default settings
    """
    try:
        # Insert new project into database
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

        # Create default settings for the project 
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
            # Rollback - Delete the project if settings creation fails
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


@router.delete("/api/projects/{project_id}")
def delete_project(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):
    """
    Delete a project and all related data
    (CASCADE handles all related data: settings, documents, chunks, chats, messages)
    """
    try:
        # Verify project exists and belongs to user
        project_result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or access denied"
            )

        # Delete project (CASCADE handles all related data) : project settings, documents, chunks will be deleted automatically
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


@router.get("/api/projects/{project_id}")
def get_project(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):
    """
    Retrieve a specific project by ID
    """
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
    

@router.get("/api/projects/{project_id}/chats")
def get_project_chats(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    """
    Retrieve all chats for a specific project
    """
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


@router.get("/api/projects/{project_id}/settings")
def get_project_settings(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):
    """
    Retrieve project settings for a specific project
    """
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




# clerk_id를 클라이언트가 서버에게 바로 넘기면, 해커가 그걸 탈취해서 delete projects api를 호출할 수도 있다. 
# 그래서 현대 앱들은 JWT 인증을 사용한다 
# Clerk도 내부적으로는 JWT 인증을 사용한다. 그래서 Clerk은 유저 회원가입 시, JWT 토큰을 주고, 브라우저는 이를 쿠키나 localStorage에 저장한다.  
# JWT 토큰은 암호화되어 있는 토큰이다. 그리고 그 안에 clerk_id를 숨겨놓았다.. 그래서 서버에서는 이를 decode할 수 있다 
# 클라이언트는 clerk_id를 바로 body에 담아서 전달하는게 아니라, authentication header에 JWT 토큰을 보내면 된다.  

# API 호출은 전부 header와 payload(body)를 가진다. header에는 token이 첨부되고, 서버는 그 토큰이 올바른지 검증한다 
# 이 API endpoint가 호출될 때마다, 검증

# 우리는 이걸 fast API method Depends() 를  사용해 할 수 있다.
# API가 호출될때, 특정 함수(clerk_id를 추출해서, 우리에게 전달해주는)를 실행시키고 싶을때  