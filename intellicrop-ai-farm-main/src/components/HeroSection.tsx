import { Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import heroFarmland from "@/assets/hero-farmland.jpg";

interface HeroSectionProps {
  onGetStarted: () => void;
}

export const HeroSection = ({ onGetStarted }: HeroSectionProps) => {
  return (
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0 z-0">
        <img 
          src={heroFarmland} 
          alt="Agricultural farmland" 
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-primary/90 via-primary/70 to-primary/95" />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 text-center animate-fade-in">
        <div className="inline-flex items-center gap-2 bg-primary-glow/20 backdrop-blur-sm border border-primary-foreground/20 rounded-full px-4 py-2 mb-6">
          <Sparkles className="w-4 h-4 text-accent" />
          <span className="text-sm font-medium text-primary-foreground">Powered by AI & Computer Vision</span>
        </div>

        <h1 className="text-5xl md:text-7xl font-bold text-primary-foreground mb-6 leading-tight">
          IntelliCrop
        </h1>
        
        <p className="text-xl md:text-2xl text-primary-foreground/90 mb-4 max-w-3xl mx-auto font-light">
          AI-Powered Crop Recommendation System
        </p>

        <p className="text-lg text-primary-foreground/80 mb-10 max-w-2xl mx-auto">
          Empowering farmers through AI, vision, and data-driven agriculture. 
          Get precise crop recommendations based on soil analysis, weather patterns, and regional data.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Button 
            size="lg" 
            onClick={onGetStarted}
            className="bg-accent text-accent-foreground hover:bg-accent/90 shadow-strong text-lg px-8 py-6 font-semibold"
          >
            Get Started
          </Button>
          <Button 
            size="lg" 
            variant="outline"
            className="bg-primary-foreground/10 text-primary-foreground border-primary-foreground/30 hover:bg-primary-foreground/20 backdrop-blur-sm text-lg px-8 py-6"
          >
            Learn More
          </Button>
        </div>

        {/* Feature Pills */}
        <div className="mt-16 flex flex-wrap gap-4 justify-center">
          {["Soil Analysis", "Weather Integration", "AI Predictions", "XAI Insights"].map((feature) => (
            <div 
              key={feature}
              className="bg-primary-foreground/10 backdrop-blur-sm border border-primary-foreground/20 rounded-full px-6 py-3 text-primary-foreground font-medium"
            >
              {feature}
            </div>
          ))}
        </div>
      </div>

      {/* Decorative Elements */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent z-10" />
    </section>
  );
};
