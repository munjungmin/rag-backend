import os
from dotenv import load_dotenv

load_dotenv()

def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"{key} must be set in .env file")
    return value

appConfig = {
    "supabase_api_url": _require("SUPABASE_API_URL"),
    "supabase_service_key": _require("SUPABASE_SERVICE_KEY"),
    "clerk_secret_key": _require("CLERK_SECRET_KEY"),

    "s3_bucket_name": _require("S3_BUCKET_NAME"),
    "aws_access_key_id": _require("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": _require("AWS_SECRET_ACCESS_KEY"),
    "aws_region": _require("AWS_REGION"),
    "aws_endpoint_url_s3": _require("AWS_ENDPOINT_URL_S3"),
    "domain": os.getenv("DOMAIN"),
    
    "openai_api_key": _require("OPENAI_API_KEY"),
    "scrapingbee_api_key": _require("SCRAPINGBEE_API_KEY"),
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
}
