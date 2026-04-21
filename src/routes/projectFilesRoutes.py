from fastapi import APIRouter, HTTPException, Depends
from src.services.supabase import supabase
from src.services.clerkAuth import get_current_user_clerk_id
from src.models.schemas import FileUploadRequest, ProcessingStatus, UrlRequest
from src.utils.util  import validate_url
from src.config.appConfig import appConfig
from src.services.awsS3 import s3_client
import uuid
from src.services.celery import perform_rag_ingestion_task


router = APIRouter(tags=["projectFilesRoutes"])

"""
`/api/projects`

  - GET `/{project_id}/files` ~ List all project files
  - POST `/{project_id}/files/upload-url` ~ Generate presigned url for file upload for frontend
  - POST `/{project_id}/files/confirm` ~ Confirmation of file upload to S3
  - POST `/{project_id}/urls` ~ Add website URL to database
  - DELETE `/{project_id}/files/{file_id}` ~ Delete document from s3 and database
  - GET `/{project_id}/files/{file_id}/chunks` ~ Get project document chunks
"""


@router.get("/{project_id}/files")
async def get_project_files(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    """
    ! 로직 흐름
    * 1. 현재 사용자 clerk_id 조회
    * 2. 주어진 project_id에 해당하는 모든 project documents 조회
    * 3. project documents 데이터 반환
    """
    try:
        project_files_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .order("created_at", desc=True)
            .execute()
        )

        # * 해당 프로젝트의 project documents가 없으면 빈 목록 반환
        # * 사용자는 project files이 있을 수도 없을 수도 있음

        return {
            "message": "Project files retrieved successfully",
            "data": project_files_result.data or [],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project {project_id} files: {str(e)}",
        )


@router.post("/{project_id}/files/upload-url")
async def get_upload_presigned_url(
    project_id: str,
    file_upload_request: FileUploadRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! 로직 흐름:
    * 1. 프로젝트가 존재하고 현재 사용자 소유인지 확인
    * 2. S3 key 생성
    * 3. 업로드용 presigned url 생성 (1시간 후 만료)
    * 4. pending 상태로 project document 레코드 생성
    * 5. presigned url 반환
    """
    try:
        # 프로젝트가 존재하고 현재 사용자 소유인지 확인
        project_ownership_verification_result = (
            supabase.table("projects")
            .select("id")
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not project_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have permission to upload files to this project",
            )

        # S3 key 생성
        file_extension = (
            file_upload_request.filename.split(".")[-1]
            if "." in file_upload_request.filename
            else ""
        )
        unique_file_id = uuid.uuid4()
        s3_key = (
            f"projects/{project_id}/documents/{unique_file_id}.{file_extension}"
            if file_extension
            else f"projects/{project_id}/documents/{unique_file_id}"
        )

        # 업로드용 presigned url 생성 (1시간 후 만료)
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": appConfig["s3_bucket_name"],
                "Key": s3_key,
                "ContentType": file_upload_request.file_type,
            },
            ExpiresIn=3600,  # 1시간
        )

        if not presigned_url:
            raise HTTPException(
                status_code=422,
                detail="Failed to generate upload presigned url",
            )

        # pending 상태로 데이터베이스 레코드 생성
        document_creation_result = (
            supabase.table("project_documents")
            .insert(
                {
                    "project_id": project_id,
                    "filename": file_upload_request.filename,
                    "s3_key": s3_key,
                    "file_size": file_upload_request.file_size,
                    "file_type": file_upload_request.file_type,
                    "processing_status": ProcessingStatus.PENDING,
                    "clerk_id": current_user_clerk_id,
                }
            )
            .execute()
        )

        if not document_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to create project document - invalid data provided",
            )

        return {
            "message": "Upload presigned url generated successfully",
            "data": {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "document": document_creation_result.data[0],
            },
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while generating upload presigned url for {project_id}: {str(e)}",
        )


@router.post("/{project_id}/files/confirm")
async def confirm_file_upload_to_s3(
    project_id: str,
    confirm_file_upload_request: dict,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! 로직 흐름:
    * 1. S3 key 제공 여부 확인
    * 2. 파일이 데이터베이스에 존재하는지 확인
    * 3. 파일 상태를 "queued"로 업데이트
    * 4. Celery - RAG Ingestion 태스크 실행
    * 5. task_id로 project document 레코드 업데이트
    * 6. 파일 업로드 완료 데이터 반환
    """
    try:
        s3_key = confirm_file_upload_request.get("s3_key")
        if not s3_key:
            raise HTTPException(
                status_code=400,
                detail="S3 key is required",
            )

        # 파일이 데이터베이스에 존재하는지 확인
        document_verification_result = (
            supabase.table("project_documents")
            .select("id")
            .eq("s3_key", s3_key)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="File not found or you don't have permission to confirm upload to S3 for this file",
            )

        # 파일 상태를 "queued"로 업데이트
        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "processing_status": ProcessingStatus.QUEUED,
                }
            )
            .eq("s3_key", s3_key)
            .execute()
        )

        # ! Celery - 백그라운드 처리 시작 - RAG Ingestion 태스크
        document_id = document_update_result.data[0]["id"]
        task_result = perform_rag_ingestion_task.delay(document_id)
        task_id = task_result.id

        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "task_id": task_id,
                }
            )
            .eq("id", document_id)
            .execute()
        )
        if not document_update_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to update project document record with task_id",
            )

        return {
            "message": "File upload to S3 confirmed successfully And Started Background Pre-Processing of this file",
            "data": document_update_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while confirming upload to S3 for {project_id}: {str(e)}",
        )


@router.post("/{project_id}/urls")
async def process_url(
    project_id: str,
    url: UrlRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! 로직 흐름:
    * 1. URL 검증
    * 2. 웹사이트 URL을 데이터베이스에 추가
    * 3. URL 백그라운드 전처리 시작
    * 4. 처리 완료된 URL 데이터 반환
    """
    try:
        # URL 검증
        url = url.url
        if url.startswith("http://") or url.startswith("https://"):
            url = url
        else:
            url = f"https://{url}"

        if not validate_url(url):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL",
            )

        # 웹사이트 URL을 데이터베이스에 추가
        document_creation_result = (
            supabase.table("project_documents")
            .insert(
                {
                    "project_id": project_id,
                    "filename": url,
                    "s3_key": "",
                    "file_size": 0,
                    "file_type": "text/html",
                    "processing_status": ProcessingStatus.QUEUED,
                    "clerk_id": current_user_clerk_id,
                    "source_type": "url",
                    "source_url": url,
                }
            )
            .execute()
        )

        if not document_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to create project document with URL Record - invalid data provided",
            )

        # ! Celery - 백그라운드 처리 시작 - RAG Ingestion 태스크
        document_id = document_creation_result.data[0]["id"]
        task_result = perform_rag_ingestion_task.delay(document_id)
        task_id = task_result.id

        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "task_id": task_id,
                }
            )
            .eq("id", document_id)
            .execute()
        )

        if not document_update_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to update project document record with task_id",
            )

        return {
            "message": "Website URL added to database successfully And Started Background Pre-Processing of this URL",
            "data": document_creation_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while processing urls for {project_id}: {str(e)}",
        )


@router.delete("/{project_id}/files/{file_id}")
async def delete_project_document(
    project_id: str,
    file_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! 로직 흐름:
    * 1. 문서가 존재하고 현재 사용자 소유인지 확인하고 전체 project document 레코드 조회
    * 2. S3에서 파일 삭제 (실제 파일에만 해당, URL 제외)
    * 3. 데이터베이스에서 문서 삭제
    * 4. 삭제 완료된 문서 데이터 반환
    """
    try:
        # 문서가 존재하고 현재 사용자 소유인지 확인하고 전체 project document 레코드 조회
        document_ownership_verification_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Document not found or you don't have permission to delete this document",
            )

        # S3에서 파일 삭제 (실제 파일에만 해당, URL 제외)
        s3_key = document_ownership_verification_result.data[0]["s3_key"]
        if s3_key:
            s3_client.delete_object(Bucket=appConfig["s3_bucket_name"], Key=s3_key)

        # 데이터베이스에서 문서 삭제
        document_deletion_result = (
            supabase.table("project_documents")
            .delete()
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_deletion_result.data:
            raise HTTPException(
                status_code=404,
                detail="Failed to delete document",
            )

        return {
            "message": "Document deleted successfully",
            "data": document_deletion_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while deleting project document {file_id} for {project_id}: {str(e)}",
        )


@router.get("/{project_id}/files/{file_id}/chunks")
async def get_project_document_chunks(
    project_id: str,
    file_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! 로직 흐름:
    * 1. 문서가 존재하고 현재 사용자 소유인지 확인하고 전체 project document 레코드 조회
    * 2. project document chunks 조회
    * 3. project document chunks 데이터 반환
    """
    try:
        # 문서가 존재하고 현재 사용자 소유인지 확인하고 전체 project document 레코드 조회
        document_ownership_verification_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Document not found or you don't have permission to delete this document",
            )

        document_chunks_result = (
            supabase.table("document_chunks")
            .select("*")
            .eq("document_id", file_id)
            .order("chunk_index")
            .execute()
        )

        return {
            "message": "Project document chunks retrieved successfully",
            "data": document_chunks_result.data or [],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while getting project document chunks for {file_id} for {project_id}: {str(e)}",
        )