from typing import Any
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import supabase, s3_client, BUCKET_NAME
from routers.auth import get_current_user

router = APIRouter(
    tags=["files"]
)

class FileUploadRequest(BaseModel):
    filename: str
    file_size: int 
    file_type: str

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


@router.post("/api/projects/{project_id}/files/upload-url")
async def get_upload_url(
    project_id: str,
    file_request: FileUploadRequest,
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Verify project exists and belongs to the user
        project_result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data:
            raise HTTPException(
                status_code=400,
                detail="Project not found or access denied"
            )
        # Generate unique S3 key 
        file_extension = file_request.filename.split('.')[-1] if '.' in file_request.filename else ''
        unique_id = str(uuid.uuid4()) 
        s3_key = f"projects/{project_id}/documents/{unique_id}.{file_extension}"     # 단순 unique_id로만 가도 됨, AWS method로 presigned_URL을 생성하기 위해서는 s3_key가 필요
        
        # Generate presigned URL (expire in 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": s3_key,
                "ContentType": file_request.file_type
            },
            ExpiresIn=3600
        )

        # Create database record with pending status 
        document_result = supabase.table("project_documents").insert({
            "project_id": project_id,
            "filename": file_request.filename,
            "s3_key": s3_key,
            "file_size": file_request.file_size,
            "file_type": file_request.file_type,
            "processing_status": "uploading",
            "clerk_id": clerk_id,
        }).execute()

        if not document_result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create document record"
            )

        return {
            "success": True,
            "message": "Upload URL generated successfully",
            "data": {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "document": document_result.data[0]
            }

        }     

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate the presigned URL: {str(e)}"
        )


@router.post("/api/projects/{project_id}/files/confirm")
async def confirm_file_upload(
    project_id: str,
    confirm_request: dict,
    clerk_id: str = Depends(get_current_user)
):
    try:
        s3_key = confirm_request.get("s3_key")

        if not s3_key:
            raise HTTPException(
                status_code=400, 
                detail="s3_key is required"
            )
        
        # Update document status
        result = supabase.table("project_documents").update({
            "processing_status": "queued"    # ready to be RAG pre-processed -> job runner에게 선택될 수 있는 상태
        }).eq("s3_key", s3_key).eq("clerk_id", clerk_id).execute()

        document = result.data[0]

        if not result.data:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # Start background preprocessing of the current file 



        # Return JSON
        return {
            "success": True,
            "message": "Upload confirmed, processing started with Celery",
            "data": document
            }
   
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to confirm upload: {str(e)}" 
        )