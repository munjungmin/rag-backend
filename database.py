from supabase import create_client, Client
from dotenv import load_dotenv
import os 

load_dotenv()

# Supabase setup
supabase_url = os.getenv("SUPABASE_API_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:    # "" None 0 [] {} 
    raise ValueError("Missing Supabase credentilas in environmen variables")

supabase: Client = create_client(supabase_url, supabase_key) # type hint (변수명: 타입힌트 = 값)
