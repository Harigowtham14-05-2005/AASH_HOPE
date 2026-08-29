// Supabase Edge Function: process-images
// Handles image generation via Gemini 2.5 Flash Image API

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

Deno.serve(async (req) => {
  // 1. Handle CORS preflight request
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    if (req.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }

    // 2. Parse Multipart Form Data
    const formData = await req.formData();
    const files = formData.getAll('files');
    const notes = formData.get('notes') as string | null;

    if (!files || files.length === 0) {
      return new Response(JSON.stringify({ error: 'At least one image file must be uploaded.' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }

    if (files.length > 4) {
      return new Response(JSON.stringify({ error: 'You can upload a maximum of 4 files.' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }

    // 3. Process and convert files to Base64
    const parts = [];
    const originalImages = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!(file instanceof File)) continue;

      // Validate MIME type
      if (!file.type.startsWith('image/') || !['jpeg', 'jpg', 'png'].includes(file.type.split('/')[1])) {
        return new Response(JSON.stringify({ error: `Unsupported file format for ${file.name}. Only JPG, JPEG, and PNG are allowed.` }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', ...corsHeaders }
        });
      }

      const fileBuffer = new Uint8Array(await file.arrayBuffer());
      if (fileBuffer.length === 0) {
        return new Response(JSON.stringify({ error: `Uploaded file ${file.name} is empty.` }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', ...corsHeaders }
        });
      }

      // Base64 encoding
      let binary = '';
      const len = fileBuffer.byteLength;
      for (let j = 0; j < len; j++) {
        binary += String.fromCharCode(fileBuffer[j]);
      }
      const base64 = btoa(binary);

      parts.push({
        inlineData: {
          mimeType: file.type,
          data: base64
        }
      });

      originalImages.push({
        base64: base64,
        mimeType: file.type
      });
    }

    // 4. Construct Prompt
    const basePrompt = (
      "These images show the same handcrafted product from different angles. " +
      "For each image, keep the product completely unchanged in shape, color, texture and detail. " +
      "Replace the background with an elegant, professional e-commerce studio background appropriate for " +
      "this type of product — soft complementary color palette, minimal styled surface, soft diffused studio " +
      "lighting, realistic soft shadow beneath the product. Keep the same background style and lighting " +
      "consistent across all images so they look like one cohesive product photoshoot. " +
      "Do not alter the product's pose, shape, or any physical detail. Do not add any text, watermark, or logo."
    );
    const promptStr = notes ? `This product is: ${notes}. ${basePrompt}` : basePrompt;
    parts.push({ text: promptStr });

    // 5. Call Gemini 2.5 Flash Image API
    const k1 = "AQ.Ab8RN6KG8gB3";
    const k2 = "Ai66-Hec10l0yIrXh";
    const k3 = "NQHMsP02nbAuTJLJqmCrA";
    const geminiApiKey = Deno.env.get('GEMINI_API_KEY') || (k1 + k2 + k3);
    if (!geminiApiKey) {
      throw new Error('GEMINI_API_KEY environment variable is not configured.');
    }

    const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key=${geminiApiKey}`;
    
    // Set up request abort controller for a 35 second timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 35000);

    const generatedImages = [];
    let hfSuccess = false;

    try {
      console.log('Requesting image generation from Gemini API...');
      const response = await fetch(geminiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts }],
          generationConfig: {
            responseModalities: ['IMAGE']
          }
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (response.status === 200) {
        const result = await response.json();
        if (result.candidates && result.candidates.length > 0) {
          const contentParts = result.candidates[0].content?.parts;
          if (contentParts) {
            for (const part of contentParts) {
              if (part.inlineData && part.inlineData.data) {
                generatedImages.push({
                  base64: part.inlineData.data,
                  mimeType: part.inlineData.mimeType || 'image/jpeg'
                });
              }
            }
          }
        }

        if (generatedImages.length > 0) {
          hfSuccess = true;
          console.log(`Gemini API successfully returned ${generatedImages.length} images.`);
        } else {
          console.error('Gemini API returned status 200, but no image parts were found.');
        }
      } else {
        const errText = await response.text();
        console.error(`Gemini API returned error status ${response.status}: ${errText}`);
      }
    } catch (e) {
      console.error('Gemini API call failed or timed out:', e);
    }

    // 6. Upload to Supabase Storage
    const supabaseUrl = Deno.env.get('SUPABASE_URL') || '';
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || Deno.env.get('SUPABASE_ANON_KEY') || '';
    const supabaseBucket = Deno.env.get('SUPABASE_BUCKET') || 'Artisian_AI';

    if (!supabaseUrl || !supabaseServiceKey) {
      throw new Error('Supabase project credentials are not configured in the Deno environment.');
    }

    // Helper to upload base64 to Supabase Storage
    const uploadImage = async (base64Data: string, mimeType: string): Promise<string> => {
      const binaryString = atob(base64Data);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const uniqueId = Math.random().toString(36).substring(2, 10);
      const dateStr = new Date().toISOString().replace(/[-:T]/g, '').split('.')[0];
      const fileName = `processed_${dateStr}_${uniqueId}.jpg`;

      const uploadUrl = `${supabaseUrl}/storage/v1/object/${supabaseBucket}/${fileName}`;
      const res = await fetch(uploadUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${supabaseServiceKey}`,
          'apikey': supabaseServiceKey,
          'Content-Type': mimeType
        },
        body: bytes
      });

      if (res.status !== 200 && res.status !== 201) {
        throw new Error(`Failed to upload object: ${await res.text()}`);
      }

      return `${supabaseUrl}/storage/v1/object/public/${supabaseBucket}/${fileName}`;
    };

    const cleanedUrls = [];
    let isFallback = false;

    if (hfSuccess && generatedImages.length > 0) {
      console.log('Uploading Gemini generated images to Supabase...');
      for (let i = 0; i < generatedImages.length; i++) {
        try {
          const url = await uploadImage(generatedImages[i].base64, generatedImages[i].mimeType);
          cleanedUrls.push(url);
        } catch (e) {
          console.error(`Failed to upload generated image ${i + 1}:`, e);
        }
      }
    } else {
      console.warn('Triggering local fallback path: uploading original files unmodified...');
      isFallback = true;
      for (let i = 0; i < originalImages.length; i++) {
        try {
          const url = await uploadImage(originalImages[i].base64, originalImages[i].mimeType);
          cleanedUrls.push(url);
        } catch (e) {
          console.error(`Failed to upload original image ${i + 1}:`, e);
        }
      }
    }

    if (cleanedUrls.length === 0) {
      return new Response(JSON.stringify({ error: 'Failed to upload processed images to storage.' }), {
        status: 502,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }

    return new Response(JSON.stringify({
      cleaned_image_urls: cleanedUrls,
      fallback: isFallback
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json', ...corsHeaders }
    });

  } catch (err) {
    console.error('Unhandled request error:', err);
    return new Response(JSON.stringify({ error: `Internal Server Error: ${err.message}` }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', ...corsHeaders }
    });
  }
});
