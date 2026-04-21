from typing import Any
from fastapi import APIRouter, HTTPException
from src.services.supabase import supabase

router = APIRouter(tags=["users"])

"""
`/api/user`

POST   /api/user/webhook    - Handle Clerk user.created webhook event
"""

@router.post("/webhook")
async def create_user_from_clerk_webhook(clerk_webhook_data: dict):
    try:
        # Step 1: webhook payload 구조 검증
        if not isinstance(clerk_webhook_data, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid webhook payload format"
            )

        # Step 2: 이벤트 타입 확인
        event_type = clerk_webhook_data.get("type")

        if event_type != "user.created":
            return {
                "success": True,
                "message": f"Event type '{event_type}' ignored"
            }

        # Step 3: 사용자 데이터 추출 및 검증
        user_data = clerk_webhook_data.get("data", {})
        if not user_data or not isinstance(user_data, dict):
            raise HTTPException(
                status_code=400,
                detail="Missing or invalid user data in webhook payload"
            )

        # Step 4: clerk_id 추출 및 검증
        clerk_id = user_data.get("id")
        if not clerk_id or not isinstance(clerk_id, str):
            raise HTTPException(
                status_code=400,
                detail="No user ID in webhook"
            )

        # Step 5: 사용자 이미 존재 여부 확인 (webhook 멱등성)
        existing_user = (
            supabase.table("users")
            .select("clerk_id")
            .eq("clerk_id", clerk_id)
            .execute()            
        )

        if existing_user.data:
            return {
                "success": True,
                "message": "User already exists",
                "clerk_id": clerk_id
            }
        
        # Step 6: 데이터베이스에 신규 사용자 생성
        result = supabase.table('users').insert({
            "clerk_id": clerk_id,
        }).execute()

        # Step 7: 삽입 성공 여부 확인
        if not result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create user in database"
            )

        return {
            "success": True,
            "message": "User created successfully",
            "data": result.data[0]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        # 예상치 못한 예외만 처리 (db 오류, 네트워크 오류 등)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while processing webhook: {str(e)}"
        )

