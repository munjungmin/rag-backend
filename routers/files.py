from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import supabase
from routers.auth import get_current_user

router = APIRouter(
    tags=["files"]
)

@router.get("/api/projects/{project_id}/files")
async def get_project_files(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Get all files for this project - FK constraints ensure projet exists and belongs to the user
        result = supabase.table("project_documents").select("*").eq("project_id", project_id).eq("clerk_id", clerk_id).order("created_at", desc=True).execute()

        return {
            "success": True,
            "message": "Project files retrieved successfully",
            "data": result.data
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get project files: {str(e)}"
        )
