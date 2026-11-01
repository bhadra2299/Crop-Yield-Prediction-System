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
    const { latitude, longitude, region } = await req.json();
    const OPENWEATHER_API_KEY = Deno.env.get('OPENWEATHER_API_KEY');

    if (!OPENWEATHER_API_KEY) {
      throw new Error('OPENWEATHER_API_KEY not configured');
    }

    console.log(`Fetching weather for coordinates: ${latitude}, ${longitude}`);

    // Fetch current weather and geocoding data in parallel
    const [weatherResponse, geoResponse] = await Promise.all([
      fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${OPENWEATHER_API_KEY}&units=metric`
      ),
      fetch(
        `https://api.openweathermap.org/geo/1.0/reverse?lat=${latitude}&lon=${longitude}&limit=1&appid=${OPENWEATHER_API_KEY}`
      )
    ]);

    if (!weatherResponse.ok) {
      throw new Error(`Weather API error: ${weatherResponse.status}`);
    }

    const weatherData = await weatherResponse.json();
    
    // Get detailed location data
    let locationData = null;
    if (geoResponse.ok) {
      const geoData = await geoResponse.json();
      locationData = geoData[0];
    }

    // India-specific state/region mapping for better agricultural data
    const indiaRegionData: Record<string, { rainfall: number; majorCrops: string[] }> = {
      'Punjab': { rainfall: 650, majorCrops: ['Wheat', 'Rice', 'Cotton'] },
      'Haryana': { rainfall: 550, majorCrops: ['Wheat', 'Rice', 'Sugarcane'] },
      'Uttar Pradesh': { rainfall: 950, majorCrops: ['Wheat', 'Rice', 'Sugarcane'] },
      'Madhya Pradesh': { rainfall: 1150, majorCrops: ['Wheat', 'Soybean', 'Gram'] },
      'Rajasthan': { rainfall: 450, majorCrops: ['Bajra', 'Wheat', 'Mustard'] },
      'Gujarat': { rainfall: 750, majorCrops: ['Cotton', 'Groundnut', 'Wheat'] },
      'Maharashtra': { rainfall: 1200, majorCrops: ['Cotton', 'Sugarcane', 'Soybean'] },
      'Karnataka': { rainfall: 1100, majorCrops: ['Rice', 'Sugarcane', 'Cotton'] },
      'Andhra Pradesh': { rainfall: 950, majorCrops: ['Rice', 'Cotton', 'Groundnut'] },
      'Telangana': { rainfall: 900, majorCrops: ['Rice', 'Cotton', 'Maize'] },
      'Tamil Nadu': { rainfall: 950, majorCrops: ['Rice', 'Sugarcane', 'Cotton'] },
      'Kerala': { rainfall: 3000, majorCrops: ['Rice', 'Coconut', 'Rubber'] },
      'West Bengal': { rainfall: 1750, majorCrops: ['Rice', 'Jute', 'Wheat'] },
      'Bihar': { rainfall: 1200, majorCrops: ['Rice', 'Wheat', 'Maize'] },
      'Odisha': { rainfall: 1500, majorCrops: ['Rice', 'Pulses', 'Oilseeds'] },
      'Assam': { rainfall: 2800, majorCrops: ['Rice', 'Tea', 'Jute'] },
      'Chhattisgarh': { rainfall: 1350, majorCrops: ['Rice', 'Maize', 'Pulses'] },
      'Jharkhand': { rainfall: 1400, majorCrops: ['Rice', 'Maize', 'Pulses'] },
      'Himachal Pradesh': { rainfall: 1600, majorCrops: ['Wheat', 'Maize', 'Rice'] },
      'Uttarakhand': { rainfall: 1550, majorCrops: ['Rice', 'Wheat', 'Sugarcane'] },
      'Jammu and Kashmir': { rainfall: 650, majorCrops: ['Rice', 'Wheat', 'Maize'] },
      'Goa': { rainfall: 3000, majorCrops: ['Rice', 'Cashew', 'Coconut'] },
    };

    // Estimate annual rainfall based on location and climate data
    const estimateAnnualRainfall = (lat: number, lon: number, weatherDesc: string, state?: string) => {
      // Check if location is in India and we have state-specific data
      if (state && indiaRegionData[state]) {
        let rainfall = indiaRegionData[state].rainfall;
        
        // Adjust based on current weather conditions
        if (weatherDesc.includes('rain')) rainfall += 150;
        if (weatherDesc.includes('cloud')) rainfall += 50;
        if (weatherDesc.includes('clear') && rainfall > 500) rainfall -= 50;
        
        return rainfall;
      }
      
      // Fallback to general climate zone estimation
      const absLat = Math.abs(lat);
      let baseRainfall = 800;
      
      // Climate zone estimation
      if (absLat < 23.5) baseRainfall = 1200; // Tropical
      else if (absLat < 35) baseRainfall = 600; // Subtropical/Arid
      else if (absLat < 60) baseRainfall = 800; // Temperate
      else baseRainfall = 400; // Polar
      
      // Adjust based on current weather
      if (weatherDesc.includes('rain')) baseRainfall += 200;
      if (weatherDesc.includes('cloud')) baseRainfall += 100;
      if (weatherDesc.includes('clear')) baseRainfall -= 100;
      
      return Math.max(200, Math.min(3500, baseRainfall));
    };

    // Extract state/region information with better India-specific detection
    const detectedState = locationData?.state || region;
    const detectedCity = weatherData.name || locationData?.name;
    const country = weatherData.sys.country;
    
    // Calculate rainfall with India-specific data if available
    const rainfall = estimateAnnualRainfall(
      latitude, 
      longitude, 
      weatherData.weather[0].description,
      country === 'IN' ? detectedState : undefined
    );

    // Extract relevant weather information
    const result = {
      temperature: Math.round(weatherData.main.temp),
      humidity: weatherData.main.humidity,
      rainfall: rainfall,
      description: weatherData.weather[0].description,
      city: detectedCity,
      region: detectedState || country,
      country: country,
      // Add India-specific agricultural context if available
      ...(country === 'IN' && detectedState && indiaRegionData[detectedState] ? {
        agriculturalZone: 'India',
        stateData: indiaRegionData[detectedState]
      } : {})
    };

    console.log('Weather data fetched successfully:', result);

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error in get-weather function:', error);
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
