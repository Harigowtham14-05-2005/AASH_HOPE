# ArtiLink Image Microservice

This microservice processes raw product photos uploaded by artisans and outputs high-quality, professionally styled, e-commerce-ready images with matching background environments using **Gemini 2.5 Flash Image**.

## Features
* **Multi-Image Cohort Processing**: Upload up to 4 angles of the same product at once, allowing the model to use context across images to output a consistent background lighting and style.
* **Dynamic Background prompts**: Generates descriptive prompts automatically from the product images and custom notes.
* **100% Demo Reliability**: Robust fallback mechanism that returns the original photos if the Gemini API fails, ensuring the endpoint never hard-crashes.
* **Memory Optimization**: Auto-downscales high-resolution images to a maximum of 1024px, preventing OOM (Out of Memory) crashes on Railway's 512MB RAM free tier.

---

## Setup & Local Installation

### 1. Get a Gemini API Key
1. Visit the [Google AI Studio API Key Manager](https://aistudio.google.com/apikey).
2. Create a free API key.

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add the following keys (see `.env.example`):
```env
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
SUPABASE_BUCKET=Artisian_AI
```

### 3. Install Dependencies
```bash
# Activate your virtual environment
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Run the Server Locally
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Documentation & Usage

Once the server is running, visit **`http://localhost:8000/docs`** to access the interactive Swagger documentation.

### 1. Health Check
* **Endpoint**: `GET /health`
* **Response**: `{"status": "ok"}`

### 2. Process Product Photos
* **Endpoint**: `POST /process-images`
* **Content-Type**: `multipart/form-data`
* **Payload**:
  * `files`: 1 to 4 raw image files (supported formats: JPG, JPEG, PNG).
  * `notes` (Optional text): Contextual notes describing the product, e.g. "handwoven silk scarf".

#### Curl Example (Single File):
```bash
curl.exe -X POST http://localhost:8000/process-images -F "files=@product.jpg" -F "notes=a colorful ceramic coffee cup"
```

#### Curl Example (Multiple Files):
```bash
curl.exe -X POST http://localhost:8000/process-images -F "files=@front.jpg" -F "files=@back.jpg" -F "files=@side.jpg" -F "notes=ceramic artisan cup"
```

#### Response:
```json
{
  "cleaned_image_urls": [
    "https://yseisxlhdffrlgosnlbh.supabase.co/storage/v1/object/public/Artisian_AI/processed_20260829_074330_609ccf77.jpg",
    "https://yseisxlhdffrlgosnlbh.supabase.co/storage/v1/object/public/Artisian_AI/processed_20260829_074332_409aef11.jpg"
  ],
  "fallback": false
}
```

---

## Free Tier Limits
The free tier of the Gemini API provides:
* **60 Requests per Minute (RPM)**
* **1,500 Requests per Day (RPD)**

This provides plenty of headroom for testing, development, and presenting on Hackathon Demo Day!
