from fastapi import APIRouter, Depends, HTTPException
from src.services.supabase import supabase
from src.models.schemas import ChatCreate
from src.services.clerkAuth import get_current_user_clerk_id

router = APIRouter(tags=["chats"])

"""
`/api/chats`

POST   /api/chats                - Create a new chat
DELETE /api/chats/{chat_id}      - Delete a chat
GET    /api/chats/{chat_id}      - Get a chat with its messages
"""


@router.post("/")
async def create_chat(
    chat: ChatCreate,
    clerk_id: str = Depends(get_current_user_clerk_id)
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


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    clerk_id: str = Depends(get_current_user_clerk_id)
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


@router.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    clerk_id: str = Depends(get_current_user_clerk_id)
):
    try:
        result = supabase.table("chats").select("*").eq("id", chat_id).eq("clerk_id", clerk_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Chat not found or access denied")

        chat = result.data[0]

        # 해당 채팅의 메시지 조회
        messages_result = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at", desc=False).execute()

        # chat 객체에 메시지 추가
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
