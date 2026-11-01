import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CloudRain, MapPin } from "lucide-react";

interface DataInputFormProps {
  weatherData: {
    temperature: string;
    rainfall: string;
    humidity: string;
  };
  regionalData: {
    region: string;
    soilPH: string;
  };
  onWeatherChange: (field: string, value: string) => void;
  onRegionalChange: (field: string, value: string) => void;
}

export const DataInputForm = ({ weatherData, regionalData, onWeatherChange, onRegionalChange }: DataInputFormProps) => {
  return (
    <div className="grid md:grid-cols-2 gap-6">
      {/* Weather Data Card */}
      <Card className="p-6 bg-gradient-card border-border shadow-medium">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 bg-primary/10 rounded-xl">
            <CloudRain className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h3 className="text-xl font-bold">Weather Data</h3>
            <p className="text-sm text-muted-foreground">Current environmental conditions</p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <Label htmlFor="temperature">Temperature (°C)</Label>
            <Input
              id="temperature"
              type="number"
              placeholder="e.g., 25"
              value={weatherData.temperature}
              onChange={(e) => onWeatherChange('temperature', e.target.value)}
              className="mt-1.5"
            />
          </div>

          <div>
            <Label htmlFor="rainfall">Annual Rainfall (mm)</Label>
            <Input
              id="rainfall"
              type="number"
              placeholder="e.g., 800"
              value={weatherData.rainfall}
              onChange={(e) => onWeatherChange('rainfall', e.target.value)}
              className="mt-1.5"
            />
          </div>

          <div>
            <Label htmlFor="humidity">Humidity (%)</Label>
            <Input
              id="humidity"
              type="number"
              placeholder="e.g., 65"
              value={weatherData.humidity}
              onChange={(e) => onWeatherChange('humidity', e.target.value)}
              className="mt-1.5"
            />
          </div>
        </div>
      </Card>

      {/* Regional Data Card */}
      <Card className="p-6 bg-gradient-card border-border shadow-medium">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 bg-primary/10 rounded-xl">
            <MapPin className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h3 className="text-xl font-bold">Regional Data</h3>
            <p className="text-sm text-muted-foreground">Location and soil properties</p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <Label htmlFor="region">Region</Label>
            <Select value={regionalData.region} onValueChange={(value) => onRegionalChange('region', value)}>
              <SelectTrigger className="mt-1.5">
                <SelectValue placeholder="Select your region" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="north">Northern Region</SelectItem>
                <SelectItem value="south">Southern Region</SelectItem>
                <SelectItem value="east">Eastern Region</SelectItem>
                <SelectItem value="west">Western Region</SelectItem>
                <SelectItem value="central">Central Region</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="soilPH">Soil pH Level</Label>
            <Input
              id="soilPH"
              type="number"
              step="0.1"
              placeholder="e.g., 6.5"
              value={regionalData.soilPH}
              onChange={(e) => onRegionalChange('soilPH', e.target.value)}
              className="mt-1.5"
            />
            <p className="text-xs text-muted-foreground mt-1.5">Range: 0 (acidic) to 14 (alkaline)</p>
          </div>

          <div className="pt-4 bg-muted/50 rounded-lg p-4 border border-border">
            <p className="text-sm font-medium mb-2">Data Quality</p>
            <div className="flex gap-2">
              <div className="h-2 flex-1 bg-success rounded-full" />
              <div className="h-2 flex-1 bg-success rounded-full" />
              <div className="h-2 flex-1 bg-muted rounded-full" />
            </div>
            <p className="text-xs text-muted-foreground mt-2">Add all fields for best results</p>
          </div>
        </div>
      </Card>
    </div>
  );
};
