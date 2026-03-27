from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os 
from routers import users, projects, files, chats

load_dotenv()

# FastAPI app 생성 
app = FastAPI(
    title="AI Engineering API",
    description="Backend API for AI Engineering application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # 내가 허용한 프론트엔드에서만 요청을 받겠다
    allow_credentials=True, # Cookie
    allow_methods=["*"],    
    allow_headers=["*"]     
)

app.include_router(users.router)
app.include_router(projects.router) 
app.include_router(files.router) 
app.include_router(chats.router) 


# Health check endpoints
@app.get("/")
async def root():
    return {"message": "AI Engineering app is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


