import { API_BASE_URL } from "@/config";

// Generic API helper
async function apiPost<T>(endpoint: string, body: Record<string, unknown>): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "API request failed");
  }
  return response.json();
}

async function apiGet<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "API request failed");
  }
  return response.json();
}

async function apiPostFile<T>(endpoint: string, file: File): Promise<T> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || "API request failed");
  }
  return response.json();
}

// ─── Prediction Response type ───────────────────────────────────
export interface PredictionResponse<T = unknown> {
  prediction: T;
  confidence?: number | null;
  metadata?: Record<string, unknown> | null;
}

// ─── Individual API calls ───────────────────────────────────────

export const healthCheck = () => apiGet<{ status: string }>("/health");

export const predictSoilImage = (file: File) =>
  apiPostFile<PredictionResponse<string>>("/predict/soil-image", file);

export const predictDisease = (file: File) =>
  apiPostFile<PredictionResponse<string>>("/predict/disease", file);

export const predictCropRecommendation = (data: {
  N: number; P: number; K: number;
  temperature: number; humidity: number; ph: number; rainfall: number;
}) => apiPost<PredictionResponse<string>>("/predict/crop-recommendation", data);

export const predictSoilNutrient = (data: {
  temperature: number; humidity: number; ph: number; rainfall: number;
}) => apiPost<PredictionResponse<{ N: number; P: number; K: number }>>("/predict/soil-nutrient", data);

export const getWeather = (lat: number, lon: number) =>
  apiGet<PredictionResponse<{ temp: number; humidity: number; condition: string }>>(
    `/predict/weather?lat=${lat}&lon=${lon}`
  );

export const predictYield = (data: {
  Crop: string; Season: string; State: string; Area: number;
  Annual_Rainfall: number; Fertilizer: number; Pesticide: number;
}) => apiPost<PredictionResponse<number>>("/predict/yield", data);

export const predictIrrigation = (data: {
  crop: string; moisture: number; temp: number;
}) => apiPost<PredictionResponse<number>>("/predict/irrigation", data);

export const predictFertilizer = (data: {
  Temperature: number; Humidity: number; Moisture: number;
  Soil_Type: string; Crop_Type: string; N: number; K: number; P: number;
}) => apiPost<PredictionResponse<string>>("/predict/fertilizer", data);

export const predictMarketPrice = (data: {
  commodity: string; forecast_horizon: number;
}) => apiPost<PredictionResponse<number>>("/predict/market-price", data);

export const predictRisk = (data: {
  yield_deviation: number; rainfall_deviation: number;
  price_volatility: number; disease_severity: number;
}) => apiPost<PredictionResponse<string>>("/predict/risk", data);

export const predictFullRecommendation = (data: {
  lat: number; lon: number; ph?: number; soil_moisture?: number;
  Area?: number; Fertilizer_used?: number; Pesticide_used?: number;
  Season?: string; State?: string;
}) => apiPost<PredictionResponse<Record<string, unknown>>>("/predict/full-recommendation", data);
