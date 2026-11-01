import { useState } from "react";
import { HeroSection } from "@/components/HeroSection";
import { SoilUpload } from "@/components/SoilUpload";
import { DataInputForm } from "@/components/DataInputForm";
import { ResultsDashboard } from "@/components/ResultsDashboard";
import { AgriChatbot } from "@/components/AgriChatbot";
import { Button } from "@/components/ui/button";
import { ArrowRight, RotateCcw, MapPin } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";

const Index = () => {
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [soilImage, setSoilImage] = useState<File | null>(null);
  const [isAutoFetching, setIsAutoFetching] = useState(false);
  
  const [weatherData, setWeatherData] = useState({
    temperature: "",
    rainfall: "",
    humidity: "",
  });

  const [regionalData, setRegionalData] = useState({
    region: "",
    soilPH: "",
  });

  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [isLoadingLocation, setIsLoadingLocation] = useState(false);
  const [soilAnalysis, setSoilAnalysis] = useState<any>(null);
  const [historicalTrends, setHistoricalTrends] = useState<any>(null);

  const handleWeatherChange = (field: string, value: string) => {
    setWeatherData(prev => ({ ...prev, [field]: value }));
  };

  const handleRegionalChange = (field: string, value: string) => {
    setRegionalData(prev => ({ ...prev, [field]: value }));
  };

  const getCurrentSeason = () => {
    const month = new Date().getMonth();
    if (month >= 2 && month <= 5) return "Summer";
    if (month >= 6 && month <= 9) return "Monsoon";
    return "Winter";
  };

  const handleGetStarted = () => {
    setShowAnalysis(true);
    setTimeout(() => {
      document.getElementById('analysis-section')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  const handleImageUpload = async (file: File) => {
    setSoilImage(file);
    setIsAutoFetching(true);
    
    // Automatically fetch location and weather data
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;
          
          try {
            const { data, error } = await supabase.functions.invoke('get-weather', {
              body: { latitude, longitude }
            });

            if (error) throw error;

            setWeatherData({
              temperature: data.temperature.toString(),
              rainfall: data.rainfall.toString(),
              humidity: data.humidity.toString(),
            });

            setRegionalData(prev => ({
              ...prev,
              region: data.region || data.city,
            }));

            toast.success(`Auto-fetched weather data for ${data.city}`);
          } catch (error) {
            console.error('Error fetching weather:', error);
            toast.error("Failed to auto-fetch weather data");
          } finally {
            setIsAutoFetching(false);
          }
        },
        (error) => {
          console.error('Geolocation error:', error);
          toast.error("Unable to auto-fetch location. Please enable location access.");
          setIsAutoFetching(false);
        }
      );
    } else {
      toast.error("Geolocation is not supported by your browser");
      setIsAutoFetching(false);
    }
  };

  const handleAnalyze = async () => {
    if (!soilImage) {
      toast.error("Please upload a soil image");
      return;
    }

    if (!weatherData.temperature || !weatherData.rainfall || !weatherData.humidity) {
      toast.error("Weather data not available. Please allow location access when uploading the image.");
      return;
    }

    if (!regionalData.region) {
      toast.error("Regional data not available. Please allow location access when uploading the image.");
      return;
    }

    toast.loading("Analyzing soil image with AI...");
    
    try {
      // Convert image to base64
      const reader = new FileReader();
      const imageBase64 = await new Promise<string>((resolve, reject) => {
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(soilImage);
      });

      // Analyze soil image with AI
      const { data: soilData, error: soilError } = await supabase.functions.invoke('analyze-soil', {
        body: { imageBase64 }
      });

      if (soilError) throw soilError;

      setSoilAnalysis(soilData);
      
      // Auto-fill soil pH from AI estimation
      if (soilData.estimatedPH) {
        setRegionalData(prev => ({
          ...prev,
          soilPH: soilData.estimatedPH.toString(),
        }));
      }
      
      toast.dismiss();
      toast.loading("Getting crop recommendations...");

      // Get crop recommendations based on soil analysis
      const { data, error } = await supabase.functions.invoke('recommend-crops', {
        body: {
          temperature: parseFloat(weatherData.temperature),
          rainfall: parseFloat(weatherData.rainfall),
          humidity: parseFloat(weatherData.humidity),
          soilPH: soilData.estimatedPH || parseFloat(regionalData.soilPH),
          soilType: soilData.soilType,
          soilCharacteristics: soilData.characteristics,
        }
      });

      if (error) throw error;

      toast.dismiss();
      toast.loading("Analyzing historical trends and seasonal predictions...");

      // Fetch historical trends and predictions
      const { data: trendsData, error: trendsError } = await supabase.functions.invoke('analyze-historical-trends', {
        body: {
          region: regionalData.region,
          currentSeason: getCurrentSeason(),
          soilType: soilData.soilType,
          temperature: parseFloat(weatherData.temperature),
          rainfall: parseFloat(weatherData.rainfall),
        }
      });

      if (trendsError) {
        console.error('Trends analysis error:', trendsError);
        // Continue without historical data
      } else {
        setHistoricalTrends(trendsData);
      }

      toast.dismiss();
      toast.success("Complete analysis ready!");
      setAnalysisResults(data);
      setShowResults(true);
      
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (error) {
      console.error('Error analyzing:', error);
      toast.dismiss();
      toast.error("Failed to analyze data. Please try again.");
    }
  };

  const handleReset = () => {
    setSoilImage(null);
    setWeatherData({ temperature: "", rainfall: "", humidity: "" });
    setRegionalData({ region: "", soilPH: "" });
    setAnalysisResults(null);
    setSoilAnalysis(null);
    setHistoricalTrends(null);
    setShowResults(false);
    setShowAnalysis(false);
    setIsAutoFetching(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    toast.info("Analysis reset");
  };

  return (
    <div className="min-h-screen">
      {/* AI Chatbot */}
      <AgriChatbot 
        userContext={{
          region: regionalData.region,
          soilType: soilAnalysis?.soilType,
          soilPH: regionalData.soilPH,
          temperature: weatherData.temperature,
          rainfall: weatherData.rainfall,
          season: getCurrentSeason(),
        }}
      />

      {/* Hero Section */}
      <HeroSection onGetStarted={handleGetStarted} />

      {/* Analysis Section */}
      {showAnalysis && (
        <section id="analysis-section" className="py-16 bg-background">
          <div className="container mx-auto px-4 max-w-6xl">
            <div className="text-center mb-12 animate-fade-in">
              <h2 className="text-4xl font-bold mb-4">Start Your Analysis</h2>
              <p className="text-xl text-muted-foreground">Upload soil data and environmental information for AI-powered recommendations</p>
            </div>

            <div className="space-y-8">
              {/* Soil Upload */}
              <SoilUpload onImageUpload={handleImageUpload} />

              {isAutoFetching && (
                <div className="text-center text-sm text-muted-foreground animate-pulse">
                  Auto-fetching location and weather data...
                </div>
              )}

              {/* Data Input Forms - Read-only, auto-populated */}
              {(weatherData.temperature || regionalData.region) && (
                <div className="space-y-4">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground">
                      ✓ Data automatically collected from your location and soil image
                    </p>
                  </div>
                  <DataInputForm
                    weatherData={weatherData}
                    regionalData={regionalData}
                    onWeatherChange={handleWeatherChange}
                    onRegionalChange={handleRegionalChange}
                  />
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4 justify-center pt-6">
                <Button
                  size="lg"
                  onClick={handleAnalyze}
                  className="bg-primary text-primary-foreground hover:bg-primary/90 shadow-medium px-8"
                >
                  Analyze & Get Recommendations
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
                
                {showResults && (
                  <Button
                    size="lg"
                    variant="outline"
                    onClick={handleReset}
                    className="border-primary text-primary hover:bg-primary/10"
                  >
                    <RotateCcw className="mr-2 w-5 h-5" />
                    Start Over
                  </Button>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Results Section */}
      {showResults && (
        <section id="results-section" className="py-16 bg-muted/30">
          <div className="container mx-auto px-4 max-w-6xl">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold mb-4">Your Crop Recommendations</h2>
              <p className="text-xl text-muted-foreground">AI-powered insights based on your soil and environmental data</p>
            </div>

            {analysisResults && (
              <ResultsDashboard
                soilType={analysisResults.soilType}
                recommendations={analysisResults.recommendations}
                xaiInsights={analysisResults.xaiInsights}
                soilAnalysis={soilAnalysis}
                historicalTrends={historicalTrends}
              />
            )}
          </div>
        </section>
      )}

      {/* Footer */}
      <footer className="py-8 bg-primary text-primary-foreground">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            © 2025 IntelliCrop - AI-Powered Crop Recommendation System
          </p>
          <p className="text-xs mt-2 opacity-80">
            Empowering farmers through AI, vision, and data-driven agriculture
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
