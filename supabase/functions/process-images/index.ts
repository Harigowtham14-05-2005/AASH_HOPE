import { encodeBase64 } from "https://deno.land/std@0.224.0/data/encode_base64.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS'
};

Deno.serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method Not Allowed' }), {
      status: 405,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }

  try {
    const formData = await req.formData();
    const files = formData.getAll('files') as File[];
    const notes = formData.get('notes') as string | null;

    if (!files || files.length === 0) {
      return new Response(JSON.stringify({ error: 'At least one image file must be uploaded.' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    if (files.length > 4) {
      return new Response(JSON.stringify({ error: 'You can upload a maximum of 4 files.' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    const fileContents: { bytes: Uint8Array, mimeType: string, filename: string }[] = [];

    for (const file of files) {
      if (!file.type.startsWith('image/')) {
        return new Response(JSON.stringify({ error: `File ${file.name} is not a valid image.` }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
      
      const fileExt = file.type.split('/').pop()?.toLowerCase();
      if (fileExt !== 'jpeg' && fileExt !== 'jpg' && fileExt !== 'png') {
        return new Response(JSON.stringify({ error: `Unsupported format for ${file.name}. Only JPG, JPEG, and PNG are allowed.` }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }

      const arrayBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);
      if (bytes.length === 0) {
        return new Response(JSON.stringify({ error: `Uploaded file ${file.name} is empty.` }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }

      fileContents.push({
        bytes,
        mimeType: file.type,
        filename: file.name
      });
    }

    // Load config variables from Supabase environment
    const geminiApiKey = Deno.env.get('GEMINI_API_KEY');
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const bucketName = Deno.env.get('SUPABASE_BUCKET') || 'Artisian_AI';

    if (!geminiApiKey) {
      console.error("Missing GEMINI_API_KEY environment variable.");
      return new Response(JSON.stringify({ error: 'Server configuration error: GEMINI_API_KEY is missing.' }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Build Gemini prompt
    const basePrompt = 
      "These images show the same handcrafted product from different angles. " +
      "For each image, keep the product completely unchanged in shape, color, texture and detail. " +
      "Replace the background with an elegant, professional e-commerce studio background appropriate for " +
      "this type of product — soft complementary color palette, minimal styled surface, soft diffused studio " +
      "lighting, realistic soft shadow beneath the product. Keep the same background style and lighting " +
      "consistent across all images so they look like one cohesive product photoshoot. " +
      "Do not alter the product's pose, shape, or any physical detail. Do not add any text, watermark, or logo.";

    const promptStr = notes ? `This product is: ${notes}. ${basePrompt}` : basePrompt;

    // Build Gemini REST API payload
    const parts = fileContents.map((file) => ({
      inlineData: {
        mimeType: file.mimeType,
        data: encodeBase64(file.bytes)
      }
    }));
    parts.push({ text: promptStr } as any);

    const payload = {
      contents: [{ parts }],
      generationConfig: {
        responseModalities: ["IMAGE"]
      }
    };

    let geminiSuccess = false;
    const generatedImages: string[] = []; // Array of base64 data

    try {
      console.log("Calling Gemini API generateContent...");
      const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key=${geminiApiKey}`;
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 35000); // 35 seconds timeout
      
      const response = await fetch(geminiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      console.log(`Gemini response status: ${response.status}`);

      if (response.status === 200) {
        const json = await response.json();
        if (json.candidates && json.candidates.length > 0) {
          const contentParts = json.candidates[0].content?.parts;
          if (contentParts) {
            for (const part of contentParts) {
              if (part.inlineData && part.inlineData.data) {
                generatedImages.push(part.inlineData.data);
              }
            }
          }
        }
        
        if (generatedImages.length > 0) {
          geminiSuccess = true;
          console.log(`Gemini API returned ${generatedImages.length} images.`);
        } else {
          console.error("Gemini API returned 200 but no image data parts were found.");
        }
      } else {
        const errText = await response.text();
        console.error(`Gemini API returned error ${response.status}: ${errText}`);
      }
    } catch (e: any) {
      console.error("Gemini API call failed or timed out:", e.message || e);
    }

    const cleanedUrls: string[] = [];
    let isFallback = false;

    // Helper function to convert base64 to Uint8Array
    const base64ToBytes = (base64: string): Uint8Array => {
      const binString = atob(base64);
      const len = binString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binString.charCodeAt(i);
      }
      return bytes;
    };

    // Helper function to upload to Supabase Storage
    const uploadFile = async (bytes: Uint8Array): Promise<string> => {
      const filePath = `processed_${Date.now()}_${Math.random().toString(36).substring(2, 10)}.jpg`;
      const uploadUrl = `${supabaseUrl}/storage/v1/object/${bucketName}/${filePath}`;
      
      const response = await fetch(uploadUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${supabaseServiceKey}`,
          'apikey': supabaseServiceKey,
          'Content-Type': 'image/jpeg'
        },
        body: bytes
      });

      if (response.status !== 200 && response.status !== 201) {
        const text = await response.text();
        throw new Error(`Upload failed (${response.status}): ${text}`);
      }
      
      return `${supabaseUrl}/storage/v1/object/public/${bucketName}/${filePath}`;
    };

    if (geminiSuccess && generatedImages.length > 0) {
      console.log("Uploading generated images to Supabase Storage...");
      for (const imgBase64 of generatedImages) {
        try {
          const bytes = base64ToBytes(imgBase64);
          const url = await uploadFile(bytes);
          cleanedUrls.push(url);
        } catch (e: any) {
          console.error("Failed to upload generated image:", e.message || e);
        }
      }
    } else {
      console.warn("Triggering fallback: uploading original images unmodified...");
      isFallback = true;
      for (const file of fileContents) {
        try {
          const url = await uploadFile(file.bytes);
          cleanedUrls.push(url);
        } catch (e: any) {
          console.error("Failed to upload fallback image:", e.message || e);
        }
      }
    }

    if (cleanedUrls.length === 0) {
      return new Response(JSON.stringify({ error: 'Failed to upload any output images.' }), {
        status: 502,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      cleaned_image_urls: cleanedUrls,
      fallback: isFallback
    }), {
      status: 200,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });

  } catch (e: any) {
    console.error("Unhandled error in function:", e);
    return new Response(JSON.stringify({ error: e.message || 'Internal Server Error' }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
});
