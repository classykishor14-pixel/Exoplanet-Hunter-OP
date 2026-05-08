import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

// Use 10.0.2.2 for Android Emulator, localhost for iOS/Web
const getBaseUrl = () => {
  if (__DEV__) {
    if (Platform.OS === 'android') {
      return 'http://10.0.2.2:8000'; // Default Android emulator IP for localhost
    }
    return 'http://localhost:8000'; // iOS simulator or web
  }
  // Production URL (replace with actual production URL later)
  return 'https://exoplanet-hunter-api.onrender.com';
};

const BASE_URL = getBaseUrl();

// Create Axios instance
const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Cache key prefix
const CACHE_PREFIX = '@ExoCache_';

// Cache configuration (24 hours)
const CACHE_EXPIRY_MS = 24 * 60 * 60 * 1000;

// --- Interfaces ---

export interface Planet {
  pl_name: string | null;
  pl_rade: number | null;
  pl_masse: number | null;
  st_rad: number | null;
  st_lum: number | null;
  pl_orbsmax: number | null;
  ra: number | null;
  dec: number | null;
}

export interface Analysis {
  is_in_habitable_zone: boolean;
  habitable_zone_inner_au: number | null;
  habitable_zone_outer_au: number | null;
  planet_type: string;
  composition_estimate: string;
  density_g_cm3: number | null;
  atmosphere_prediction: string;
}

export interface SearchResult {
  planet: string;
  analysis: Analysis;
  raw_data: Planet;
}

export interface NASAImage {
  title: string;
  description: string;
  image_url: string;
}

export interface Recommendation {
  name: string;
  host_star: string;
  type_emoji: string;
  category: string;
  description: string;
  thumbnail_url: string;
}

// --- Helper Functions ---

// Get data from cache if it exists and is not expired
const getCachedData = async <T>(key: string): Promise<T | null> => {
  try {
    const cachedItem = await AsyncStorage.getItem(CACHE_PREFIX + key);
    if (cachedItem) {
      const { data, timestamp } = JSON.parse(cachedItem);
      if (Date.now() - timestamp < CACHE_EXPIRY_MS) {
        return data as T;
      } else {
        // Expired
        await AsyncStorage.removeItem(CACHE_PREFIX + key);
      }
    }
  } catch (error) {
    console.error('Error reading from cache', error);
  }
  return null;
};

// Save data to cache
const setCachedData = async <T>(key: string, data: T): Promise<void> => {
  try {
    const cacheItem = JSON.stringify({
      data,
      timestamp: Date.now(),
    });
    await AsyncStorage.setItem(CACHE_PREFIX + key, cacheItem);
  } catch (error) {
    console.error('Error writing to cache', error);
  }
};

// --- API Methods ---

export const getRecommendations = async (): Promise<Recommendation[]> => {
  const cacheKey = 'recommendations';
  const cached = await getCachedData<Recommendation[]>(cacheKey);
  
  if (cached) {
    return cached;
  }

  try {
    const response = await apiClient.get<{ recommendations: Recommendation[] }>('/api/recommendations');
    const data = response.data.recommendations;
    await setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    // If network fails, try to return expired cache as fallback
    try {
      const cachedItem = await AsyncStorage.getItem(CACHE_PREFIX + cacheKey);
      if (cachedItem) {
        return JSON.parse(cachedItem).data as Recommendation[];
      }
    } catch (e) {
      // Ignore
    }
    throw error;
  }
};

export const searchPlanet = async (query: string): Promise<Planet> => {
  try {
    const response = await apiClient.get<Planet>(`/api/search`, {
      params: { q: query },
    });
    return response.data;
  } catch (error) {
    console.error(`Error searching planet ${query}:`, error);
    throw error;
  }
};

export const autocompletePlanets = async (query: string): Promise<string[]> => {
  if (query.length < 2) return [];
  try {
    const response = await apiClient.get<{ suggestions: string[] }>(`/api/search`, {
      params: { q: query },
    });
    return response.data.suggestions || [];
  } catch (error) {
    console.error(`Error fetching autocomplete for ${query}:`, error);
    return [];
  }
};

export const getPlanetDetails = async (name: string): Promise<SearchResult> => {
  const cacheKey = `details_${name}`;
  const cached = await getCachedData<SearchResult>(cacheKey);
  
  if (cached) {
    return cached;
  }

  try {
    const response = await apiClient.get<SearchResult>(`/api/planet/${encodeURIComponent(name)}/details`);
    const data = response.data;
    await setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    console.error(`Error fetching planet details for ${name}:`, error);
    // Fallback to expired cache
    try {
      const cachedItem = await AsyncStorage.getItem(CACHE_PREFIX + cacheKey);
      if (cachedItem) {
        return JSON.parse(cachedItem).data as SearchResult;
      }
    } catch (e) {
      // Ignore
    }
    throw error;
  }
};

export const getPlanetImages = async (name: string): Promise<NASAImage[]> => {
  const cacheKey = `images_${name}`;
  const cached = await getCachedData<NASAImage[]>(cacheKey);
  
  if (cached) {
    return cached;
  }

  try {
    const response = await apiClient.get<{ images: NASAImage[] }>(`/api/planet/${encodeURIComponent(name)}/images`);
    const data = response.data.images;
    await setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    console.error(`Error fetching images for ${name}:`, error);
    // Fallback to expired cache
    try {
      const cachedItem = await AsyncStorage.getItem(CACHE_PREFIX + cacheKey);
      if (cachedItem) {
        return JSON.parse(cachedItem).data as NASAImage[];
      }
    } catch (e) {
      // Ignore
    }
    throw error;
  }
};

export default {
  getRecommendations,
  searchPlanet,
  getPlanetDetails,
  getPlanetImages,
};
