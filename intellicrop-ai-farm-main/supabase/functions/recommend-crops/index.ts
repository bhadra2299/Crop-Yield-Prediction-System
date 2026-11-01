import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface CropData {
  name: string;
  confidence: number;
  suitability: string;
  expectedYield: string;
  season: string;
  minTemp: number;
  maxTemp: number;
  minPH: number;
  maxPH: number;
  minRainfall: number;
  maxRainfall: number;
}

const cropDatabase: CropData[] = [
  {
    name: "Rice",
    confidence: 0,
    suitability: "",
    expectedYield: "4.5-6.0 tons/ha",
    season: "Monsoon",
    minTemp: 20,
    maxTemp: 35,
    minPH: 5.5,
    maxPH: 7.0,
    minRainfall: 1000,
    maxRainfall: 2500,
  },
  {
    name: "Wheat",
    confidence: 0,
    suitability: "",
    expectedYield: "4.0-5.0 tons/ha",
    season: "Winter",
    minTemp: 10,
    maxTemp: 25,
    minPH: 6.0,
    maxPH: 7.5,
    minRainfall: 400,
    maxRainfall: 800,
  },
  {
    name: "Corn",
    confidence: 0,
    suitability: "",
    expectedYield: "5.5-7.0 tons/ha",
    season: "Summer",
    minTemp: 18,
    maxTemp: 32,
    minPH: 5.8,
    maxPH: 7.0,
    minRainfall: 500,
    maxRainfall: 1000,
  },
  {
    name: "Cotton",
    confidence: 0,
    suitability: "",
    expectedYield: "2.5-3.5 tons/ha",
    season: "Summer",
    minTemp: 21,
    maxTemp: 35,
    minPH: 5.8,
    maxPH: 8.0,
    minRainfall: 500,
    maxRainfall: 1000,
  },
  {
    name: "Soybeans",
    confidence: 0,
    suitability: "",
    expectedYield: "2.5-3.5 tons/ha",
    season: "Summer",
    minTemp: 20,
    maxTemp: 30,
    minPH: 6.0,
    maxPH: 7.5,
    minRainfall: 450,
    maxRainfall: 900,
  },
  {
    name: "Sugarcane",
    confidence: 0,
    suitability: "",
    expectedYield: "60-80 tons/ha",
    season: "Year-round",
    minTemp: 20,
    maxTemp: 35,
    minPH: 6.0,
    maxPH: 7.5,
    minRainfall: 1000,
    maxRainfall: 2500,
  },
  {
    name: "Barley",
    confidence: 0,
    suitability: "",
    expectedYield: "3.0-4.0 tons/ha",
    season: "Winter",
    minTemp: 10,
    maxTemp: 25,
    minPH: 6.5,
    maxPH: 7.8,
    minRainfall: 300,
    maxRainfall: 650,
  },
  {
    name: "Chickpeas",
    confidence: 0,
    suitability: "",
    expectedYield: "1.5-2.5 tons/ha",
    season: "Winter",
    minTemp: 15,
    maxTemp: 30,
    minPH: 6.0,
    maxPH: 7.5,
    minRainfall: 400,
    maxRainfall: 700,
  },
];

function calculateCropSuitability(
  crop: CropData,
  temperature: number,
  rainfall: number,
  humidity: number,
  soilPH: number
): CropData {
  let score = 0;
  let factors = 0;

  // Temperature score
  if (temperature >= crop.minTemp && temperature <= crop.maxTemp) {
    const tempOptimal = (crop.minTemp + crop.maxTemp) / 2;
    const tempDeviation = Math.abs(temperature - tempOptimal);
    const tempRange = (crop.maxTemp - crop.minTemp) / 2;
    score += Math.max(0, 100 - (tempDeviation / tempRange) * 50);
    factors++;
  }

  // Rainfall score
  if (rainfall >= crop.minRainfall && rainfall <= crop.maxRainfall) {
    const rainOptimal = (crop.minRainfall + crop.maxRainfall) / 2;
    const rainDeviation = Math.abs(rainfall - rainOptimal);
    const rainRange = (crop.maxRainfall - crop.minRainfall) / 2;
    score += Math.max(0, 100 - (rainDeviation / rainRange) * 50);
    factors++;
  }

  // pH score
  if (soilPH >= crop.minPH && soilPH <= crop.maxPH) {
    const phOptimal = (crop.minPH + crop.maxPH) / 2;
    const phDeviation = Math.abs(soilPH - phOptimal);
    const phRange = (crop.maxPH - crop.minPH) / 2;
    score += Math.max(0, 100 - (phDeviation / phRange) * 50);
    factors++;
  }

  // Humidity consideration (general bonus)
  if (humidity >= 50 && humidity <= 80) {
    score += 10;
    factors += 0.1;
  }

  const confidence = factors > 0 ? Math.round(score / factors) : 0;
  
  let suitability = "Poor";
  if (confidence >= 80) suitability = "Excellent";
  else if (confidence >= 65) suitability = "Good";
  else if (confidence >= 50) suitability = "Moderate";

  return {
    ...crop,
    confidence,
    suitability,
  };
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { temperature, rainfall, humidity, soilPH, soilType } = await req.json();

    console.log('Analyzing crop recommendations with data:', {
      temperature,
      rainfall,
      humidity,
      soilPH,
      soilType,
    });

    // Calculate suitability for each crop
    const recommendations = cropDatabase
      .map(crop => calculateCropSuitability(crop, temperature, rainfall, humidity, soilPH))
      .filter(crop => crop.confidence > 0)
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 4);

    // Generate XAI insights
    const xaiInsights = [
      `The ${soilType || 'analyzed soil'} shows favorable characteristics for the recommended crops.`,
      `Current temperature (${temperature}°C) and rainfall (${rainfall}mm) align well with ${recommendations[0]?.name || 'optimal'} growing conditions.`,
      `Soil pH level (${soilPH}) indicates ${soilPH >= 6.0 && soilPH <= 7.5 ? 'neutral to slightly acidic' : soilPH < 6.0 ? 'acidic' : 'alkaline'} soil, which supports nutrient availability for most recommended crops.`,
      `Humidity level at ${humidity}% ${humidity >= 50 && humidity <= 80 ? 'is optimal' : humidity < 50 ? 'is low, consider irrigation' : 'is high, monitor for diseases'} for crop growth.`,
    ];

    const result = {
      soilType: soilType || "Analyzed Soil",
      recommendations,
      xaiInsights,
    };

    console.log('Crop recommendations generated:', result);

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error in recommend-crops function:', error);
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
