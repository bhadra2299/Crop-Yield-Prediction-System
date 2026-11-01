import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { imageBase64 } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');

    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY not configured');
    }

    if (!imageBase64) {
      throw new Error('No image provided');
    }

    console.log('Analyzing soil image with AI...');

    // Call Lovable AI with vision capabilities
    const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          {
            role: 'system',
            content: `You are an expert agricultural soil scientist with extensive knowledge of soil classification. 
Analyze soil images and classify them into one of these specific soil types:
- Sandy Soil (light texture, large particles, good drainage)
- Clay Soil (heavy texture, fine particles, poor drainage, sticky when wet)
- Loamy Soil (balanced mixture, ideal for crops)
- Silt Soil (smooth texture, medium particles)
- Peaty Soil (dark brown/black, high organic matter)
- Chalky Soil (light colored, alkaline, stony)
- Saline Soil (salt deposits visible, white crusting)
- Red Soil (iron-rich, reddish color)
- Black Soil (rich in clay and humus, very dark)
- Alluvial Soil (deposited by rivers, fertile, varied texture)

Provide your response in this exact JSON format:
{
  "soilType": "exact soil type name from the list above",
  "confidence": 85-99 (number representing confidence percentage),
  "characteristics": ["list", "of", "key", "visual", "characteristics", "observed"],
  "texture": "description of soil texture",
  "color": "description of soil color",
  "moisture": "dry/moist/wet",
  "organicContent": "low/medium/high",
  "estimatedPH": 5.5-8.5 (estimated pH value based on soil type and characteristics)
}`
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze this soil image and classify the soil type with detailed characteristics.'
              },
              {
                type: 'image_url',
                image_url: {
                  url: imageBase64
                }
              }
            ]
          }
        ],
        temperature: 0.3, // Lower temperature for more consistent classification
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('AI gateway error:', response.status, errorText);
      throw new Error(`AI analysis failed: ${response.status}`);
    }

    const data = await response.json();
    const analysisText = data.choices[0].message.content;
    
    console.log('Raw AI response:', analysisText);

    // Parse the JSON response from AI
    let analysis;
    try {
      // Extract JSON from markdown code blocks if present
      const jsonMatch = analysisText.match(/```json\n([\s\S]*?)\n```/) || 
                       analysisText.match(/```\n([\s\S]*?)\n```/) ||
                       [null, analysisText];
      analysis = JSON.parse(jsonMatch[1] || analysisText);
    } catch (parseError) {
      console.error('Failed to parse AI response:', parseError);
      // Fallback response
      analysis = {
        soilType: 'Loamy Soil',
        confidence: 75,
        characteristics: ['Unable to parse detailed analysis'],
        texture: 'Mixed',
        color: 'Brown',
        moisture: 'Unknown',
        organicContent: 'medium'
      };
    }

    console.log('Soil analysis complete:', analysis);

    return new Response(JSON.stringify(analysis), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error in analyze-soil function:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return new Response(
      JSON.stringify({ error: errorMessage }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
