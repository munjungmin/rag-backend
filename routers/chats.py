from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import supabase
from routers.auth import get_current_user
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Initialize LLM for summarisation
llm = ChatOpenAI(model="gpt-4o", temperature=0)

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

class SendMessageRequest(BaseModel):
    content: str

@router.post("/api/projects/{project_id}/chats/{chat_id}/messages")
async def send_message(
    chat_id: str,
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


        # Call LLM with system prompt + user message
        print(f"🤖 Calling LLM...")
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Provide clear, concise, and accurate"),
            HumanMessage(content=message)
        ]

        response = llm.invoke(messages)
        ai_response = response.content

        print(f"✅ LLM response received: {len(ai_response)} chars")


        # 3. Save AI message
        print(f"💾 Saving AI message...")
        ai_message_result = supabase.table("messages").insert({
            "chat_id": chat_id,
            "content": ai_response,
            "role": "assistant",
            "clerk_id": clerk_id,
            "citations": []
        }).execute()

        ai_message = ai_message_result.data[0]
        print(f"✅ AI message saved: {ai_message['id']}")

        # 4. Return data
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
