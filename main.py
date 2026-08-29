import io
import uuid
import base64
import logging
import datetime
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from google import genai
from google.genai import types

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("image-service")

app = FastAPI(
    title="ArtiLink Image Microservice",
    description="Microservice to process product photos and regenerate backgrounds using Gemini 2.5 Flash Image.",
    version="2.0.0"
)

# Enable permissive CORS for hackathon cross-origin calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


def upload_to_supabase(image_bytes: bytes) -> str:
    """Uploads image bytes to Supabase Storage and returns the public CDN URL."""
    now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    file_path = f"processed_{now}_{unique_id}.jpg"

    supabase_upload_url = f"{config.SUPABASE_URL.rstrip('/')}/storage/v1/object/{config.SUPABASE_BUCKET}/{file_path}"
    supabase_headers = {
        "Authorization": f"Bearer {config.SUPABASE_SERVICE_KEY}",
        "apikey": config.SUPABASE_SERVICE_KEY,
        "Content-Type": "image/jpeg"
    }

    logger.info(f"Uploading image to Supabase path: {file_path}")
    response = requests.post(
        supabase_upload_url,
        headers=supabase_headers,
        data=image_bytes,
        timeout=15
    )

    if response.status_code not in (200, 201):
        logger.error(f"Supabase upload returned status {response.status_code}: {response.text}")
        raise Exception(f"Failed to upload image to Supabase: {response.text}")
        
    public_url = f"{config.SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{config.SUPABASE_BUCKET}/{file_path}"
    logger.info(f"Successfully uploaded image to Supabase. CDN URL: {public_url}")
    return public_url


def run_image_pipeline(file_contents: list[bytes], notes: str) -> dict:
    """
    Synchronous pipeline function that processes the images using Gemini 2.5 Flash Image.
    Executed in a background thread pool by FastAPI to prevent blocking the event loop.
    """
    # 1. Validate environment configuration
    missing_vars = config.validate_config()
    if missing_vars:
        logger.error(f"Configuration validation failed: missing {missing_vars}")
        raise HTTPException(
            status_code=500,
            detail=f"Server configuration error. Missing settings: {', '.join(missing_vars)}"
        )
    
    # 2. Process and downscale input images for performance/RAM stability
    logger.info(f"Step 2: Processing {len(file_contents)} input images...")
    pil_images = []
    for idx, content in enumerate(file_contents):
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            
            # Stability/RAM optimization: Downscale high-resolution images to a maximum of 1024px
            # to prevent Out of Memory (OOM) crashes on Railway's 512MB RAM free tier.
            max_dimension = 1024
            if max(img.size) > max_dimension:
                logger.info(f"Resizing input image {idx+1} from {img.size} to max {max_dimension}px for stability...")
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
            pil_images.append(img)
        except Exception as e:
            logger.error(f"Failed to open/resize image {idx+1}: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read file at index {idx+1}. Make sure it is a valid JPG/PNG image."
            )

    # 3. Build the prompt dynamically
    base_prompt = (
        "These images show the same handcrafted product from different angles. "
        "For each image, keep the product completely unchanged in shape, color, texture and detail. "
        "Replace the background with an elegant, professional e-commerce studio background appropriate for "
        "this type of product — soft complementary color palette, minimal styled surface, soft diffused studio "
        "lighting, realistic soft shadow beneath the product. Keep the same background style and lighting "
        "consistent across all images so they look like one cohesive product photoshoot. "
        "Do not alter the product's pose, shape, or any physical detail. Do not add any text, watermark, or logo."
    )
    if notes:
        prompt_str = f"This product is: {notes}. {base_prompt}"
    else:
        prompt_str = base_prompt

    # 4. Call Gemini 2.5 Flash Image API
    logger.info("Step 3: Initializing Gemini Client and calling API...")
    # Initialize genai client with 35 seconds request timeout
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=35_000)
    )

    contents_payload = pil_images + [prompt_str]
    logger.info(f"Sending prompt: '{prompt_str}'")

    generated_images = []
    hf_success = False

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents_payload,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )
        
        # Parse output images from parts
        if response.candidates and len(response.candidates) > 0:
            content = response.candidates[0].content
            if content and content.parts:
                for part in content.parts:
                    if part.inline_data and part.inline_data.data:
                        generated_images.append(part.inline_data.data)
                        
        if len(generated_images) > 0:
            hf_success = True
            logger.info(f"Gemini API returned {len(generated_images)} generated images successfully.")
            if len(generated_images) < len(file_contents):
                logger.warning(
                    f"Gemini API returned fewer images ({len(generated_images)}) than were uploaded ({len(file_contents)})."
                )
        else:
            logger.error("Gemini API call succeeded but returned 0 images in the response parts.")
            
    except Exception as e:
        logger.error(f"Gemini API call failed or timed out: {str(e)}")

    # 5. Upload to Supabase (with fallback handling)
    cleaned_urls = []
    is_fallback = False

    if hf_success and len(generated_images) > 0:
        logger.info("Step 4: Uploading generated images to Supabase Storage...")
        for idx, img_bytes in enumerate(generated_images):
            try:
                url = upload_to_supabase(img_bytes)
                cleaned_urls.append(url)
                logger.info(f"Uploaded generated image {idx+1}/{len(generated_images)} successfully.")
            except Exception as e:
                logger.error(f"Failed to upload generated image {idx+1}: {str(e)}")
    else:
        # Fallback path: Upload original images unmodified
        logger.warning("Triggering local fallback: uploading original images unmodified to Supabase...")
        is_fallback = True
        for idx, img_bytes in enumerate(file_contents):
            try:
                url = upload_to_supabase(img_bytes)
                cleaned_urls.append(url)
                logger.info(f"Uploaded fallback original image {idx+1}/{len(file_contents)} successfully.")
            except Exception as e:
                logger.error(f"Failed to upload fallback image {idx+1}: {str(e)}")

    # Ensure we return at least an empty list if all uploads failed, but raise 502 if nothing could upload
    if len(cleaned_urls) == 0:
        logger.error("Failed to upload any images (generated or fallback) to Supabase Storage.")
        raise HTTPException(
            status_code=502,
            detail="Failed to upload output images to cloud storage."
        )

    return {
        "cleaned_image_urls": cleaned_urls,
        "fallback": is_fallback
    }


@app.post("/process-images")
async def process_images(
    files: list[UploadFile] = File(...),
    notes: str = Form(None)
):
    """
    Main endpoint accepting 1 to 4 raw product photos and optional notes.
    Returns JSON containing array of cleaned_image_urls.
    """
    logger.info("Received request on /process-images")
    
    # 1. Validation
    if not files or len(files) == 0:
        logger.error("Request missing upload files")
        raise HTTPException(status_code=400, detail="At least one image file must be uploaded.")

    if len(files) > 4:
        logger.error(f"Uploaded {len(files)} files, which exceeds the limit of 4")
        raise HTTPException(status_code=400, detail="You can upload a maximum of 4 files.")

    file_contents = []
    for file in files:
        if not file.filename:
            continue
            
        # Check supported file types (JPG, JPEG, PNG)
        content_type = file.content_type
        if not content_type or not content_type.startswith("image/"):
            logger.error(f"File {file.filename} is not an image.")
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image.")
            
        file_ext = content_type.split("/")[-1].lower()
        if file_ext not in ("jpeg", "jpg", "png"):
            logger.error(f"Unsupported file format: {file_ext} for file {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format for {file.filename}. Only JPG, JPEG, and PNG are allowed."
            )

        content = await file.read()
        if len(content) == 0:
            logger.error(f"Uploaded file {file.filename} is empty")
            raise HTTPException(status_code=400, detail=f"Uploaded file {file.filename} is empty.")
            
        file_contents.append(content)

    if len(file_contents) == 0:
        logger.error("No valid upload files found in request")
        raise HTTPException(status_code=400, detail="At least one valid image file must be uploaded.")

    logger.info(f"Processing {len(file_contents)} files successfully validated. Starting background threadpool...")
    
    # Execute the heavy blocking pipeline in FastAPI's external thread pool
    try:
        result = await run_in_threadpool(run_image_pipeline, file_contents, notes)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Unhandled error in processing pipeline")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
