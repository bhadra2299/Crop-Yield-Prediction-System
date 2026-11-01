import { useState, useCallback } from "react";
import { Upload, Image as ImageIcon, X, CheckCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

interface SoilUploadProps {
  onImageUpload: (file: File) => void;
}

export const SoilUpload = ({ onImageUpload }: SoilUploadProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      toast.error("Please upload an image file");
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    onImageUpload(file);
    toast.success("Soil image uploaded successfully!");
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  return (
    <Card className="p-8 bg-gradient-card border-border shadow-medium">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-primary/10 rounded-xl">
          <ImageIcon className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-foreground">Soil Image Analysis</h2>
          <p className="text-muted-foreground">Upload a clear image of your soil for AI analysis</p>
        </div>
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`relative border-2 border-dashed rounded-2xl p-8 transition-all duration-300 ${
          isDragging 
            ? "border-primary bg-primary/5 scale-[1.02]" 
            : "border-border hover:border-primary/50 bg-muted/30"
        }`}
      >
        {!preview ? (
          <div className="text-center">
            <div className="mx-auto w-20 h-20 bg-primary/10 rounded-2xl flex items-center justify-center mb-4">
              <Upload className="w-10 h-10 text-primary" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Drop your soil image here</h3>
            <p className="text-muted-foreground mb-4">or click to browse from your device</p>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              className="hidden"
              id="soil-upload"
            />
            <label htmlFor="soil-upload">
              <Button asChild>
                <span className="cursor-pointer">Choose File</span>
              </Button>
            </label>
            <p className="text-sm text-muted-foreground mt-4">Supports: JPG, PNG, WEBP</p>
          </div>
        ) : (
          <div className="relative">
            <img 
              src={preview} 
              alt="Soil preview" 
              className="w-full h-64 object-cover rounded-xl shadow-soft"
            />
            <Button
              size="icon"
              variant="destructive"
              className="absolute top-4 right-4 shadow-medium"
              onClick={() => {
                setPreview(null);
                toast.info("Image removed");
              }}
            >
              <X className="w-4 h-4" />
            </Button>
            <div className="absolute bottom-4 left-4 bg-success text-success-foreground px-4 py-2 rounded-lg flex items-center gap-2 shadow-medium">
              <CheckCircle className="w-4 h-4" />
              <span className="font-medium">Image Ready</span>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};
