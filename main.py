import io
import uuid
import base64
import logging
import datetime
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, ImageDraw, ImageFilter
# pyrefly: ignore [missing-import]
import rembg

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("image-service")

# Initialize rembg session once at module load (startup)
logger.info("Initializing rembg (u2net) model session...")
try:
    rembg_session = rembg.new_session("u2net")
    logger.info("rembg session initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize rembg session: {str(e)}")
    # We don't crash here so that health check and startup can still proceed
    rembg_session = None

app = FastAPI(
    title="ArtiLink Image Microservice",
    description="Microservice to isolate products and regenerate background using AI inpainting.",
    version="1.0.0"
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


def create_vertical_gradient(size, color_start, color_end):
    """
    Creates a vertical gradient image transitioning from color_start (top)
    to color_end (bottom) as a professional studio cyclorama backdrop.
    """
    width, height = size
    # Create a 1D gradient line
    gradient = Image.new("L", (1, height))
    for y in range(height):
        # Interpolate from 0 (start) to 255 (end)
        val = int(y * 255 / (height - 1) if height > 1 else 0)
        gradient.putpixel((0, y), val)
    
    # Resize to fill the full width of the image
    gradient = gradient.resize((width, height))
    
    # Create start and end color images
    start_img = Image.new("RGB", (width, height), color_start)
    end_img = Image.new("RGB", (width, height), color_end)
    
    # Composite them using the gradient as the mask
    return Image.composite(end_img, start_img, gradient)


def run_image_pipeline(file_bytes: bytes, prompt: str) -> dict:
    """
    Synchronous pipeline function that processes the image.
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
    
    if rembg_session is None:
        logger.error("rembg session is not initialized.")
        raise HTTPException(
            status_code=500,
            detail="Background removal engine is not initialized."
        )

    # 2. Isolate product cutout
    logger.info("Step 2: Running rembg to isolate product...")
    try:
        input_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        # remove background, producing an RGBA image
        rgba_image = rembg.remove(input_image, session=rembg_session)
    except Exception as e:
        logger.exception("rembg execution failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to isolate product from background: {str(e)}"
        )

    # 3. Build inpainting mask (invert alpha) and feather cutout edges
    logger.info("Step 3: Creating inverted inpainting mask and feathering cutout edges...")
    try:
        # Extract sharp original alpha channel
        alpha = rgba_image.split()[3]
        
        # Apply a slight Gaussian blur (radius 2px) to the alpha channel edges for soft blending
        feathered_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Merge the feathered alpha channel back into rgba_image
        r, g, b, _ = rgba_image.split()
        rgba_image = Image.merge("RGBA", (r, g, b, feathered_alpha))
        
        # Build inverted mask from original sharp alpha channel (white = inpaint, black = keep product)
        mask_image = ImageOps.invert(alpha)
    except Exception as e:
        logger.exception("Failed to build mask from alpha channel")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate processing mask: {str(e)}"
        )

    # 4. Generate Image Caption using Salesforce/blip-image-captioning-large
    caption = None
    logger.info("Step 4a: Calling Hugging Face for image captioning...")
    try:
        caption_headers = {
            "Authorization": f"Bearer {config.HF_API_TOKEN}"
        }
        caption_response = requests.post(
            "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-large",
            headers=caption_headers,
            data=file_bytes,
            timeout=10
        )
        logger.info(f"Image captioning API response status code: {caption_response.status_code}")
        
        if caption_response.status_code == 200:
            result = caption_response.json()
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                caption = result[0]["generated_text"].strip()
                logger.info(f"Successfully generated image caption: '{caption}'")
            else:
                logger.warning(f"Unexpected image captioning response format: {result}")
        else:
            try:
                cap_error = caption_response.json()
            except Exception:
                cap_error = caption_response.text
            logger.warning(
                f"Image captioning API returned status {caption_response.status_code}: {cap_error}. "
                "Gracefully falling back to generic e-commerce prompt."
            )
    except Exception as e:
        logger.warning(
            f"Image captioning API call failed or timed out: {str(e)}. "
            "Gracefully falling back to generic e-commerce prompt."
        )

    # 5. Request background regeneration from Hugging Face Inference API
    logger.info("Step 4b: Requesting background regeneration from Hugging Face Inference API...")
    
    # Construct prompts
    if prompt:
        prompt_str = prompt
        logger.info(f"Using user-provided custom prompt: '{prompt_str}'")
    else:
        if caption:
            prompt_str = f"professional e-commerce studio background suited for {caption}, soft complementary color palette, elegant minimal styled surface, soft diffused studio lighting, subtle realistic shadow, photorealistic, 8k, sharp focus on product, no text, no watermark, no logo"
            logger.info(f"Using dynamically generated prompt: '{prompt_str}'")
        else:
            prompt_str = "soft neutral gradient studio background, professional e-commerce lighting, photorealistic, subtle shadow"
            logger.info(f"Using generic fallback prompt: '{prompt_str}'")

    negative_prompt_str = (
        "flat plain white, harsh lighting, blown out highlights, cartoon, illustration, "
        "blurry, low quality, distorted, extra objects, text, watermark, logo"
    )
    logger.info(f"Final Inpainting Prompt: '{prompt_str}'")
    logger.info(f"Shared Negative Prompt: '{negative_prompt_str}'")

    # Prepare images as Base64 encoded PNGs for Hugging Face REST API
    original_buffered = io.BytesIO()
    input_image.save(original_buffered, format="PNG")
    original_b64 = base64.b64encode(original_buffered.getvalue()).decode("utf-8")

    mask_buffered = io.BytesIO()
    mask_image.save(mask_buffered, format="PNG")
    mask_b64 = base64.b64encode(mask_buffered.getvalue()).decode("utf-8")

    hf_headers = {
        "Authorization": f"Bearer {config.HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    hf_payload = {
        "inputs": prompt_str,
        "image": original_b64,
        "mask_image": mask_b64,
        "parameters": {
            "negative_prompt": negative_prompt_str
        }
    }

    hf_success = False
    inpainted_bg = None

    try:
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2-inpainting",
            headers=hf_headers,
            json=hf_payload,
            timeout=15
        )
        
        logger.info(f"Hugging Face inpainting API response status code: {response.status_code}")
        
        if response.status_code == 200:
            inpainted_bg = Image.open(io.BytesIO(response.content)).convert("RGB")
            hf_success = True
            logger.info("Hugging Face inpainting call SUCCEEDED. AI background generated successfully.")
        else:
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text
                
            logger.error(
                f"Hugging Face inpainting API FAILED (Status: {response.status_code}). Error body: {error_body}. "
                "Triggering fallback light-grey-to-white gradient background."
            )
    except Exception as e:
        logger.error(
            f"Hugging Face inpainting API request failed or timed out: {str(e)}. "
            "Triggering fallback light-grey-to-white gradient background."
        )

    # 6. Composite original cutout back on top with drop shadow
    logger.info("Step 5: Compositing isolated product cutout onto background...")
    try:
        # Determine base background
        if hf_success and inpainted_bg is not None:
            # Resize background if dimensions do not match the input
            if inpainted_bg.size != input_image.size:
                logger.info(f"Resizing generated background from {inpainted_bg.size} to {input_image.size}")
                inpainted_bg = inpainted_bg.resize(input_image.size, Image.Resampling.LANCZOS)
            base_bg = inpainted_bg.copy()
        else:
            # Fallback to nice soft light-grey-to-white gradient background
            logger.info("Applying fallback: creating soft light-grey-to-white gradient background...")
            base_bg = create_vertical_gradient(
                input_image.size,
                (255, 255, 255),  # Top: White
                (230, 230, 230)   # Bottom: Soft light-grey
            )

        # Generate soft elliptical drop shadow layer beneath the product
        # Get bounding box of the cutout using the original alpha channel
        bbox = alpha.getbbox()
        if bbox:
            left, top, right, bottom = bbox
            width_box = right - left
            center_x = (left + right) // 2
            
            # Shadow dimensions relative to the product width
            shadow_w = int(width_box * 0.90)  # slightly narrower than product
            shadow_h = int(width_box * 0.15)  # elliptical height (vertical thickness)
            
            # Bounding box coordinates for the shadow ellipse at the bottom of the product
            shadow_box = (
                center_x - shadow_w // 2,
                bottom - shadow_h // 2,
                center_x + shadow_w // 2,
                bottom + shadow_h // 2
            )
            
            logger.info(f"Generating soft elliptical drop shadow at bounding box: {shadow_box}")
            
            # Create a transparent overlay for the shadow
            shadow_layer = Image.new("RGBA", base_bg.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(shadow_layer)
            
            # Draw a semi-transparent dark grey ellipse
            shadow_color = (40, 40, 40, 110)
            draw.ellipse(shadow_box, fill=shadow_color)
            
            # Blur the shadow to make it soft and realistic
            blur_radius = max(3, shadow_h // 3)
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Paste the shadow layer onto the background layer
            base_bg.paste(shadow_layer, (0, 0), mask=shadow_layer)
            logger.info("Soft drop shadow successfully applied beneath product position.")
        else:
            logger.warning("Product alpha channel has no bounding box. Skipping shadow.")

        # Composite the product cutout (rgba_image with feathered alpha) on top
        final_image = base_bg.convert("RGB")
        final_image.paste(rgba_image, (0, 0), mask=rgba_image)
        logger.info("Successfully composited product cutout onto final image.")
        
    except Exception as e:
        logger.exception("Failed to composite product cutout onto background")
        raise HTTPException(
            status_code=500,
            detail=f"Image composition failed: {str(e)}"
        )

    # 7. Upload final JPEG to Supabase Storage via REST
    logger.info("Step 6: Converting composite to JPEG and uploading to Supabase Storage...")
    try:
        jpeg_buffered = io.BytesIO()
        final_image.save(jpeg_buffered, format="JPEG", quality=90)
        jpeg_bytes = jpeg_buffered.getvalue()

        # Generate unique path using timestamp and random string
        now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        file_path = f"processed_{now}_{unique_id}.jpg"

        supabase_upload_url = f"{config.SUPABASE_URL.rstrip('/')}/storage/v1/object/{config.SUPABASE_BUCKET}/{file_path}"
        supabase_headers = {
            "Authorization": f"Bearer {config.SUPABASE_SERVICE_KEY}",
            "apikey": config.SUPABASE_SERVICE_KEY,
            "Content-Type": "image/jpeg"
        }

        upload_response = requests.post(
            supabase_upload_url,
            headers=supabase_headers,
            data=jpeg_bytes,
            timeout=15
        )

        if upload_response.status_code not in (200, 201):
            logger.error(f"Supabase upload returned status {upload_response.status_code}: {upload_response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to upload image to Supabase: {upload_response.text}"
            )
        logger.info("Uploaded processed image to Supabase Storage.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Failed to upload output image to Supabase")
        raise HTTPException(
            status_code=502,
            detail=f"Supabase Storage connection failed: {str(e)}"
        )

    # 8. Return public URL
    public_url = f"{config.SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{config.SUPABASE_BUCKET}/{file_path}"
    logger.info(f"Pipeline successful. Returning public URL: {public_url}")
    return {"cleaned_image_url": public_url}


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form(None)
):
    """
    Main endpoint accepting raw product photo and prompt.
    Returns JSON containing the cleaned_image_url.
    """
    logger.info("Received request on /process-image")
    
    # Validate request payload
    if not file or not file.filename:
        logger.error("Request missing upload file")
        raise HTTPException(status_code=400, detail="Empty upload file.")

    if not file.content_type.startswith("image/"):
        logger.error(f"Invalid file content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        logger.error("Uploaded file has zero bytes")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Execute the heavy blocking pipeline in FastAPI's external thread pool
    try:
        result = await run_in_threadpool(run_image_pipeline, file_bytes, prompt)
        return result
    except HTTPException as he:
        # Re-raise standard HTTP exceptions
        raise he
    except Exception as e:
        logger.exception("Unhandled error in processing pipeline")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
