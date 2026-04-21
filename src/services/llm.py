from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config.appConfig import appConfig

openAI = {
    "embeddings": OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=appConfig["openai_api_key"],
        dimensions=1536,  # ! 변경하면 안되는 값: document_chunks 임베딩 벡터에서 사용
    ),
    "chat_llm": ChatOpenAI(
        model="gpt-4o", api_key=appConfig["openai_api_key"], temperature=0
    ),
}