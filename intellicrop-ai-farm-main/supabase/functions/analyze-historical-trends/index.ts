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
    const { region, currentSeason, soilType, temperature, rainfall } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');

    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY not configured');
    }

    console.log(`Analyzing historical crop trends for region: ${region}`);

    // India-specific agricultural context for better predictions
    const indiaStateData: Record<string, { majorCrops: string[], avgYield: Record<string, string>, marketTrends: string }> = {
      'Punjab': { 
        majorCrops: ['Wheat', 'Rice', 'Cotton', 'Maize', 'Sugarcane'],
        avgYield: { 'Wheat': '4.8 tons/ha', 'Rice': '4.2 tons/ha', 'Cotton': '2.5 tons/ha' },
        marketTrends: 'Strong MSP support, high mechanization, export quality grain production'
      },
      'Haryana': { 
        majorCrops: ['Wheat', 'Rice', 'Cotton', 'Bajra', 'Mustard'],
        avgYield: { 'Wheat': '4.5 tons/ha', 'Rice': '3.8 tons/ha', 'Cotton': '2.3 tons/ha' },
        marketTrends: 'MSP procurement, proximity to Delhi market advantage'
      },
      'Uttar Pradesh': { 
        majorCrops: ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Potato'],
        avgYield: { 'Wheat': '3.2 tons/ha', 'Sugarcane': '68 tons/ha', 'Rice': '2.8 tons/ha' },
        marketTrends: 'Large market, diverse zones, sugarcane belt with mill support'
      },
      'Maharashtra': { 
        majorCrops: ['Cotton', 'Sugarcane', 'Soybean', 'Jowar', 'Wheat'],
        avgYield: { 'Cotton': '1.8 tons/ha', 'Sugarcane': '85 tons/ha', 'Soybean': '1.2 tons/ha' },
        marketTrends: 'Cotton hub, strong cooperative sugar mills, emerging organic farming'
      },
      'Gujarat': { 
        majorCrops: ['Cotton', 'Groundnut', 'Wheat', 'Cumin', 'Castor'],
        avgYield: { 'Cotton': '2.2 tons/ha', 'Groundnut': '2.0 tons/ha', 'Wheat': '3.5 tons/ha' },
        marketTrends: 'Cotton export hub, spice trading, good irrigation infrastructure'
      },
      'Madhya Pradesh': { 
        majorCrops: ['Soybean', 'Wheat', 'Gram', 'Maize', 'Cotton'],
        avgYield: { 'Soybean': '1.3 tons/ha', 'Wheat': '2.8 tons/ha', 'Gram': '1.0 tons/ha' },
        marketTrends: 'Soybean capital of India, growing pulse production, increasing irrigation'
      },
      'Karnataka': { 
        majorCrops: ['Rice', 'Sugarcane', 'Cotton', 'Maize', 'Ragi'],
        avgYield: { 'Rice': '3.5 tons/ha', 'Sugarcane': '75 tons/ha', 'Cotton': '2.0 tons/ha' },
        marketTrends: 'Progressive farming, IT integration, organic farming growth'
      },
      'Andhra Pradesh': { 
        majorCrops: ['Rice', 'Cotton', 'Chili', 'Groundnut', 'Turmeric'],
        avgYield: { 'Rice': '3.8 tons/ha', 'Cotton': '2.1 tons/ha', 'Chili': '2.5 tons/ha' },
        marketTrends: 'Major rice exporter, spice hub, aquaculture integration'
      },
      'Telangana': { 
        majorCrops: ['Rice', 'Cotton', 'Maize', 'Turmeric', 'Chili'],
        avgYield: { 'Rice': '3.6 tons/ha', 'Cotton': '2.0 tons/ha', 'Maize': '6.5 tons/ha' },
        marketTrends: 'Irrigation expansion, technology adoption, horticulture growth'
      },
      'Tamil Nadu': { 
        majorCrops: ['Rice', 'Sugarcane', 'Cotton', 'Groundnut', 'Coconut'],
        avgYield: { 'Rice': '4.0 tons/ha', 'Sugarcane': '95 tons/ha', 'Cotton': '1.9 tons/ha' },
        marketTrends: 'Delta agriculture, water-efficient farming, export-oriented'
      },
      'West Bengal': { 
        majorCrops: ['Rice', 'Jute', 'Wheat', 'Potato', 'Mustard'],
        avgYield: { 'Rice': '2.8 tons/ha', 'Jute': '2.5 tons/ha', 'Potato': '25 tons/ha' },
        marketTrends: 'Multiple cropping, jute industry, vegetable production'
      },
      'Rajasthan': { 
        majorCrops: ['Bajra', 'Wheat', 'Mustard', 'Barley', 'Gram'],
        avgYield: { 'Bajra': '1.5 tons/ha', 'Wheat': '2.5 tons/ha', 'Mustard': '1.2 tons/ha' },
        marketTrends: 'Dryland farming, oilseed focus, micro-irrigation adoption'
      }
    };

    const isIndianState = Object.keys(indiaStateData).some(state => region.includes(state));
    const stateKey = Object.keys(indiaStateData).find(state => region.includes(state));
    const stateInfo = stateKey ? indiaStateData[stateKey] : null;

    const systemPrompt = `You are an agricultural data analyst with access to comprehensive historical crop data spanning the last 10 years from India and global agricultural databases. 
Analyze agricultural trends for the given region, soil type, and climate conditions.

${stateInfo ? `SPECIFIC DATA FOR ${stateKey}:
- Major crops: ${stateInfo.majorCrops.join(', ')}
- Average yields: ${Object.entries(stateInfo.avgYield).map(([crop, cropYield]) => `${crop}: ${cropYield}`).join(', ')}
- Market context: ${stateInfo.marketTrends}` : ''}

Provide your response in this EXACT JSON format:
{
  "historicalTrends": {
    "mostGrownCrops": [
      {
        "name": "Crop name",
        "percentage": 35,
        "trend": "increasing/stable/decreasing",
        "averageYield": "5.5 tons/ha"
      }
    ],
    "mostProfitableCrops": [
      {
        "name": "Crop name",
        "avgProfitPerHa": "$2500",
        "roi": "180%",
        "marketDemand": "high/medium/low",
        "priceStability": "stable/volatile"
      }
    ]
  },
  "seasonalPredictions": {
    "nextSeason": "Summer/Monsoon/Winter",
    "predictedWeather": {
      "temperature": "25-32°C",
      "rainfall": "600-800mm",
      "conditions": "favorable/moderate/challenging"
    },
    "recommendedCrops": [
      {
        "name": "Crop name",
        "successRate": 85,
        "reason": "Brief explanation based on 10-year patterns"
      }
    ]
  },
  "tenYearInsights": [
    "Insight 1: Specific trend observed over 10 years",
    "Insight 2: Climate pattern affecting crop selection",
    "Insight 3: Market trend observation"
  ]
}`;

    const userPrompt = `Analyze historical agricultural data for:
Region: ${region}
Current Season: ${currentSeason}
Soil Type: ${soilType}
Current Temperature: ${temperature}°C
Current Rainfall: ${rainfall}mm/year

${isIndianState ? `
INDIA-SPECIFIC CONTEXT:
- Government MSP (Minimum Support Price) available for wheat, rice, pulses
- Major agricultural markets: APMCs, e-NAM digital platform
- Growing demand for organic certification and export quality
- Increasing adoption of precision agriculture and drip irrigation
- Climate challenges: monsoon variability, water stress in western regions
- Opportunity crops: oilseeds (reducing import dependency), pulses, horticulture
${stateInfo ? `\n- Local strengths for ${stateKey}: ${stateInfo.marketTrends}` : ''}
` : ''}

Based on 10-year historical patterns, agricultural research data, and climate trends for this region:
1. Identify the top 5 most grown crops in this region over the last 10 years with accurate local context
2. Identify the top 5 most profitable crops based on market data, ROI, and local pricing trends
3. Predict suitable crops for the next growing season considering seasonal weather patterns
4. Provide data-driven insights about crop trends, climate impacts, market dynamics, and government policies

${isIndianState ? 'Emphasize India-specific factors like MSP, export potential, water efficiency, and local market demand.' : 'Use real agricultural patterns for similar regions.'}`;


    const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        temperature: 0.3,
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

    let analysis;
    try {
      const jsonMatch = analysisText.match(/```json\n([\s\S]*?)\n```/) || 
                       analysisText.match(/```\n([\s\S]*?)\n```/) ||
                       [null, analysisText];
      analysis = JSON.parse(jsonMatch[1] || analysisText);
    } catch (parseError) {
      console.error('Failed to parse AI response:', parseError);
      analysis = {
        historicalTrends: {
          mostGrownCrops: [],
          mostProfitableCrops: []
        },
        seasonalPredictions: {
          nextSeason: currentSeason,
          predictedWeather: { temperature: "N/A", rainfall: "N/A", conditions: "moderate" },
          recommendedCrops: []
        },
        tenYearInsights: ["Historical data analysis unavailable"]
      };
    }

    console.log('Historical analysis complete');

    return new Response(JSON.stringify(analysis), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error in analyze-historical-trends function:', error);
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
