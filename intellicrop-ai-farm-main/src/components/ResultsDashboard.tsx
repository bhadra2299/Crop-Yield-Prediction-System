import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Sprout, TrendingUp, Info, Sparkles, DollarSign, Calendar, LineChart, Target } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface CropRecommendation {
  name: string;
  confidence: number;
  suitability: string;
  expectedYield: string;
  season: string;
}

interface SoilAnalysis {
  soilType: string;
  confidence: number;
  characteristics: string[];
  texture: string;
  color: string;
  moisture: string;
  organicContent: string;
}

interface HistoricalTrends {
  historicalTrends: {
    mostGrownCrops: Array<{
      name: string;
      percentage: number;
      trend: string;
      averageYield: string;
    }>;
    mostProfitableCrops: Array<{
      name: string;
      avgProfitPerHa: string;
      roi: string;
      marketDemand: string;
      priceStability: string;
    }>;
  };
  seasonalPredictions: {
    nextSeason: string;
    predictedWeather: {
      temperature: string;
      rainfall: string;
      conditions: string;
    };
    recommendedCrops: Array<{
      name: string;
      successRate: number;
      reason: string;
    }>;
  };
  tenYearInsights: string[];
}

interface ResultsDashboardProps {
  soilType: string;
  recommendations: CropRecommendation[];
  xaiInsights: string[];
  soilAnalysis?: SoilAnalysis;
  historicalTrends?: HistoricalTrends;
}

export const ResultsDashboard = ({ soilType, recommendations, xaiInsights, soilAnalysis, historicalTrends }: ResultsDashboardProps) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-success";
    if (confidence >= 60) return "text-accent";
    return "text-muted-foreground";
  };

  const getSuitabilityColor = (suitability: string) => {
    if (suitability === "Excellent") return "bg-success text-success-foreground";
    if (suitability === "Good") return "bg-primary text-primary-foreground";
    return "bg-secondary text-secondary-foreground";
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Soil Analysis Result */}
      <Card className="p-6 bg-gradient-hero border-primary shadow-strong">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <div className="p-4 bg-primary-foreground/20 rounded-2xl backdrop-blur-sm">
              <Sparkles className="w-8 h-8 text-primary-foreground" />
            </div>
            <div>
              <p className="text-sm text-primary-foreground/80 font-medium">Detected Soil Type</p>
              <h3 className="text-3xl font-bold text-primary-foreground">{soilType}</h3>
            </div>
          </div>
          <Badge className="bg-accent text-accent-foreground text-sm px-4 py-2">AI Analyzed</Badge>
        </div>

        {soilAnalysis && (
          <>
            <div className="mb-4">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-primary-foreground/80">AI Confidence</span>
                <span className="font-semibold text-primary-foreground">{soilAnalysis.confidence}%</span>
              </div>
              <Progress value={soilAnalysis.confidence} className="h-2 bg-primary-foreground/20" />
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                <p className="text-xs text-primary-foreground/70 mb-1">Texture</p>
                <p className="font-medium text-primary-foreground">{soilAnalysis.texture}</p>
              </div>
              <div className="p-3 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                <p className="text-xs text-primary-foreground/70 mb-1">Color</p>
                <p className="font-medium text-primary-foreground">{soilAnalysis.color}</p>
              </div>
              <div className="p-3 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                <p className="text-xs text-primary-foreground/70 mb-1">Moisture</p>
                <p className="font-medium text-primary-foreground capitalize">{soilAnalysis.moisture}</p>
              </div>
              <div className="p-3 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                <p className="text-xs text-primary-foreground/70 mb-1">Organic Content</p>
                <p className="font-medium text-primary-foreground capitalize">{soilAnalysis.organicContent}</p>
              </div>
            </div>

            {soilAnalysis.characteristics.length > 0 && (
              <div className="p-4 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                <p className="text-sm font-medium text-primary-foreground mb-2">Key Characteristics</p>
                <ul className="list-disc list-inside space-y-1 text-sm text-primary-foreground/80">
                  {soilAnalysis.characteristics.map((char, idx) => (
                    <li key={idx}>{char}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </Card>

      {/* Crop Recommendations */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <Sprout className="w-6 h-6 text-primary" />
          <h3 className="text-2xl font-bold">Recommended Crops</h3>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          {recommendations.map((crop, index) => (
            <Card key={index} className="p-5 bg-gradient-card border-border shadow-medium hover:shadow-strong transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h4 className="text-xl font-bold mb-1">{crop.name}</h4>
                  <Badge className={getSuitabilityColor(crop.suitability)}>
                    {crop.suitability}
                  </Badge>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-primary">{crop.confidence}%</p>
                  <p className="text-xs text-muted-foreground">Confidence</p>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-muted-foreground">Confidence Score</span>
                    <span className={`font-semibold ${getConfidenceColor(crop.confidence)}`}>
                      {crop.confidence}%
                    </span>
                  </div>
                  <Progress value={crop.confidence} className="h-2" />
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border">
                  <div>
                    <p className="text-xs text-muted-foreground">Expected Yield</p>
                    <p className="font-semibold text-foreground">{crop.expectedYield}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Best Season</p>
                    <p className="font-semibold text-foreground">{crop.season}</p>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* XAI Insights */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <TrendingUp className="w-6 h-6 text-primary" />
          <h3 className="text-2xl font-bold">Explainable AI Insights</h3>
        </div>

        <div className="space-y-3">
          {xaiInsights.map((insight, index) => (
            <Alert key={index} className="bg-gradient-card border-primary/20">
              <Info className="h-4 w-4 text-primary" />
              <AlertDescription className="text-foreground ml-2">
                {insight}
              </AlertDescription>
            </Alert>
          ))}
        </div>
      </div>

      {/* Historical Trends - Last 10 Years */}
      {historicalTrends && (
        <>
          <div className="space-y-6">
            <div className="flex items-center gap-3">
              <LineChart className="w-6 h-6 text-primary" />
              <h3 className="text-2xl font-bold">Historical Crop Analysis (10 Years)</h3>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              {/* Most Grown Crops */}
              <Card className="p-6 bg-gradient-card border-border shadow-medium">
                <h4 className="font-bold text-lg mb-4 flex items-center gap-2">
                  <Sprout className="w-5 h-5 text-primary" />
                  Most Grown Crops in Region
                </h4>
                <div className="space-y-4">
                  {historicalTrends.historicalTrends.mostGrownCrops.map((crop, idx) => (
                    <div key={idx} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold">{crop.name}</span>
                        <div className="flex items-center gap-2">
                          <Badge variant={crop.trend === 'increasing' ? 'default' : 'secondary'} className="text-xs">
                            {crop.trend}
                          </Badge>
                          <span className="font-bold text-primary">{crop.percentage}%</span>
                        </div>
                      </div>
                      <Progress value={crop.percentage} className="h-2" />
                      <p className="text-xs text-muted-foreground">Avg Yield: {crop.averageYield}</p>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Most Profitable Crops */}
              <Card className="p-6 bg-gradient-card border-border shadow-medium">
                <h4 className="font-bold text-lg mb-4 flex items-center gap-2">
                  <DollarSign className="w-5 h-5 text-success" />
                  Most Profitable Crops
                </h4>
                <div className="space-y-4">
                  {historicalTrends.historicalTrends.mostProfitableCrops.map((crop, idx) => (
                    <div key={idx} className="p-3 bg-success/5 rounded-lg border border-success/20">
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-semibold text-foreground">{crop.name}</span>
                        <Badge className="bg-success text-success-foreground">{crop.roi} ROI</Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <p className="text-muted-foreground">Profit/Ha</p>
                          <p className="font-bold text-success">{crop.avgProfitPerHa}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Market Demand</p>
                          <p className="font-semibold capitalize">{crop.marketDemand}</p>
                        </div>
                        <div className="col-span-2">
                          <p className="text-muted-foreground">Price Stability</p>
                          <p className="font-semibold capitalize">{crop.priceStability}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>

          {/* Seasonal Predictions */}
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Calendar className="w-6 h-6 text-primary" />
              <h3 className="text-2xl font-bold">Next Season Predictions</h3>
            </div>

            <Card className="p-6 bg-gradient-hero border-primary shadow-strong">
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-bold text-xl text-primary-foreground">
                    {historicalTrends.seasonalPredictions.nextSeason} Season
                  </h4>
                  <Badge className={`${
                    historicalTrends.seasonalPredictions.predictedWeather.conditions === 'favorable' 
                      ? 'bg-success' 
                      : 'bg-accent'
                  } text-white`}>
                    {historicalTrends.seasonalPredictions.predictedWeather.conditions}
                  </Badge>
                </div>
                <div className="grid grid-cols-3 gap-3 mt-3">
                  <div className="p-3 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                    <p className="text-xs text-primary-foreground/70 mb-1">Temperature</p>
                    <p className="font-semibold text-primary-foreground">
                      {historicalTrends.seasonalPredictions.predictedWeather.temperature}
                    </p>
                  </div>
                  <div className="p-3 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                    <p className="text-xs text-primary-foreground/70 mb-1">Rainfall</p>
                    <p className="font-semibold text-primary-foreground">
                      {historicalTrends.seasonalPredictions.predictedWeather.rainfall}
                    </p>
                  </div>
                  <div className="p-3 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                    <p className="text-xs text-primary-foreground/70 mb-1">Conditions</p>
                    <p className="font-semibold text-primary-foreground capitalize">
                      {historicalTrends.seasonalPredictions.predictedWeather.conditions}
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <h5 className="font-semibold text-primary-foreground flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Recommended Crops for Next Season
                </h5>
                {historicalTrends.seasonalPredictions.recommendedCrops.map((crop, idx) => (
                  <div key={idx} className="p-4 bg-primary-foreground/10 rounded-lg backdrop-blur-sm">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-bold text-primary-foreground">{crop.name}</span>
                      <span className="text-sm font-semibold text-success">{crop.successRate}% Success Rate</span>
                    </div>
                    <p className="text-sm text-primary-foreground/80">{crop.reason}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* 10-Year Insights */}
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="w-6 h-6 text-primary" />
              <h3 className="text-2xl font-bold">Data-Driven Insights (10 Years)</h3>
            </div>
            <div className="space-y-3">
              {historicalTrends.tenYearInsights.map((insight, idx) => (
                <Alert key={idx} className="bg-gradient-card border-primary/20">
                  <Info className="h-4 w-4 text-primary" />
                  <AlertDescription className="text-foreground ml-2">
                    {insight}
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Action Recommendations */}
      <Card className="p-6 bg-accent/10 border-accent/30">
        <h4 className="font-bold text-lg mb-3 flex items-center gap-2">
          <Sprout className="w-5 h-5 text-accent" />
          Next Steps
        </h4>
        <ul className="space-y-2 text-sm">
          <li className="flex items-start gap-2">
            <span className="text-accent mt-0.5">•</span>
            <span>Prepare soil with organic matter for optimal nutrient retention</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-accent mt-0.5">•</span>
            <span>Consider crop rotation to maintain soil health</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-accent mt-0.5">•</span>
            <span>Monitor weather patterns for optimal planting time</span>
          </li>
          {historicalTrends && (
            <li className="flex items-start gap-2">
              <span className="text-accent mt-0.5">•</span>
              <span>Review historical profitability data when selecting crops for maximum ROI</span>
            </li>
          )}
        </ul>
      </Card>
    </div>
  );
};
