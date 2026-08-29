import os
import logging
from dotenv import load_dotenv

# Set up logging for the config module
logger = logging.getLogger("image-service.config")

# Load variables from .env if present
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

# Log warnings for missing configurations on startup, but do not raise an error yet
# This allows the FastAPI app to start up and run health checks
def validate_config():
    missing = []
    if not HF_API_TOKEN:
        missing.append("HF_API_TOKEN")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_KEY:
        missing.append("SUPABASE_SERVICE_KEY")
    if not SUPABASE_BUCKET:
        missing.append("SUPABASE_BUCKET")
    return missing

missing_vars = validate_config()
if missing_vars:
    logger.warning(
        f"Missing configuration variables: {', '.join(missing_vars)}. "
        "The /process-image endpoint will fail with HTTP 500 until these are provided."
    )
else:
    logger.info("All configuration environment variables loaded successfully.")
