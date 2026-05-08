from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import math
import asyncio

app = FastAPI(
    title="Exoplanet Hunter Backend",
    description="Backend API for the Exoplanet Hunter mobile app",
    version="1.0.0"
)

# Allow CORS for mobile app and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
NASA_IMAGES_API_URL = "https://images-api.nasa.gov/search"

# Hardcoded famous exoplanets list
FAMOUS_EXOPLANETS = [
    {"name": "Kepler-442 b", "host_star": "Kepler-442", "type_emoji": "🌍", "category": "Habitable", "description": "A super-Earth-sized exoplanet orbiting a K-type main-sequence star. It is considered one of the most habitable exoplanets discovered.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/kepler-442b.jpg"},
    {"name": "TRAPPIST-1 e", "host_star": "TRAPPIST-1", "type_emoji": "🌍", "category": "Habitable", "description": "An Earth-sized exoplanet orbiting an ultra-cool dwarf star. It is located in the habitable zone and is a prime target for studying exoplanet atmospheres.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/trappist-1e.jpg"},
    {"name": "Proxima Centauri b", "host_star": "Proxima Centauri", "type_emoji": "🌍", "category": "Habitable", "description": "The closest known exoplanet to the Solar System. It orbits within the habitable zone of the red dwarf Proxima Centauri.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/proxima-b.jpg"},
    {"name": "Kepler-186 f", "host_star": "Kepler-186", "type_emoji": "🌍", "category": "Habitable", "description": "The first Earth-sized exoplanet discovered in the habitable zone of another star.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/kepler-186f.jpg"},
    {"name": "K2-18 b", "host_star": "K2-18", "type_emoji": "🌍", "category": "Habitable", "description": "A super-Earth exoplanet orbiting a red dwarf. Water vapor was detected in its atmosphere, making it a significant target of interest.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/k2-18b.jpg"},
    {"name": "LHS 1140 b", "host_star": "LHS 1140", "type_emoji": "🌍", "category": "Habitable", "description": "A massive, dense super-Earth located in the habitable zone of its host star. It is an excellent candidate for atmospheric studies.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/lhs-1140b.jpg"},
    {"name": "Teegarden's Star b", "host_star": "Teegarden's Star", "type_emoji": "🌍", "category": "Habitable", "description": "An Earth-mass exoplanet orbiting within the habitable zone of Teegarden's Star, an ultra-cool M-type dwarf.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/teegarden-b.jpg"},
    {"name": "Ross 128 b", "host_star": "Ross 128", "type_emoji": "🌍", "category": "Habitable", "description": "An Earth-sized exoplanet orbiting a quiet red dwarf star, located relatively close to our solar system.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/ross-128b.jpg"},
    {"name": "Gliese 667 C c", "host_star": "Gliese 667 C", "type_emoji": "🌍", "category": "Habitable", "description": "A super-Earth orbiting within the habitable zone of a triple star system.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/gliese-667cc.jpg"},
    {"name": "Kepler-62 f", "host_star": "Kepler-62", "type_emoji": "🌍", "category": "Habitable", "description": "A super-Earth exoplanet orbiting within the habitable zone of the star Kepler-62. It may be a water world.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/kepler-62f.jpg"},
    {"name": "WASP-12 b", "host_star": "WASP-12", "type_emoji": "🪐", "category": "Gas Giants", "description": "A hot Jupiter that is being stretched into an egg shape and slowly consumed by its host star.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/wasp-12b.jpg"},
    {"name": "51 Pegasi b", "host_star": "51 Pegasi", "type_emoji": "🪐", "category": "Gas Giants", "description": "The first exoplanet discovered orbiting a Sun-like star, establishing the existence of 'hot Jupiters'.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/51-pegasi-b.jpg"},
    {"name": "HD 209458 b", "host_star": "HD 209458", "type_emoji": "🪐", "category": "Gas Giants", "description": "Often called 'Osiris', it was the first exoplanet seen to transit its star and the first to have its atmosphere detected.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/hd-209458b.jpg"},
    {"name": "Kepler-10 b", "host_star": "Kepler-10", "type_emoji": "🪨", "category": "Rocky", "description": "The first unambiguously rocky exoplanet discovered by the Kepler Space Telescope.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/kepler-10b.jpg"},
    {"name": "TRAPPIST-1 d", "host_star": "TRAPPIST-1", "type_emoji": "🪨", "category": "Rocky", "description": "Another Earth-sized planet in the remarkable TRAPPIST-1 system, also located near or in the habitable zone.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/trappist-1d.jpg"},
    {"name": "Kelt-9 b", "host_star": "Kelt-9", "type_emoji": "🌋", "category": "Lava Worlds", "description": "The hottest known exoplanet, hotter than many stars, with a dayside temperature exceeding 4,300 degrees Celsius.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/kelt-9b.jpg"},
    {"name": "GJ 1214 b", "host_star": "GJ 1214", "type_emoji": "🪐", "category": "Gas Giants", "description": "A classic 'water world' candidate, a super-Earth with a thick, steamy atmosphere.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/gj-1214b.jpg"},
    {"name": "WASP-107 b", "host_star": "WASP-107", "type_emoji": "🪐", "category": "Gas Giants", "description": "A 'super-puff' planet with a mass similar to Neptune but a size approaching Jupiter, making it incredibly low density.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/wasp-107b.jpg"},
    {"name": "TOI-700 d", "host_star": "TOI-700", "type_emoji": "🌍", "category": "Habitable", "description": "The first Earth-sized habitable-zone planet discovered by the TESS mission.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/toi-700d.jpg"},
    {"name": "Kepler-22 b", "host_star": "Kepler-22", "type_emoji": "🌍", "category": "Habitable", "description": "The first confirmed planet by Kepler to orbit in the habitable zone of a Sun-like star.", "thumbnail_url": "https://exoplanet-hunter-assets.s3.amazonaws.com/thumbnails/kepler-22b.jpg"}
]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Exoplanet Hunter API is running!"}

@app.get("/api/search")
async def search_planet(q: str = Query(..., description="Star or planet name")):
    """
    Search NASA Exoplanet Archive TAP (pscomppars table) for planet data.
    If query is short (<5 chars), return autocomplete suggestions.
    Otherwise, return full planet data.
    """
    query_str = q.strip().replace("'", "''")
    
    # If query is short, return autocomplete suggestions
    if len(query_str) < 5:
        adql = f"SELECT DISTINCT pl_name FROM pscomppars WHERE LOWER(pl_name) LIKE LOWER('{query_str}%') LIMIT 10"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    NASA_TAP_URL,
                    params={"query": adql, "format": "json"},
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    suggestions = [row["pl_name"] for row in data if row["pl_name"]]
                    return {"suggestions": suggestions}
            except Exception as e:
                print(f"Error querying NASA TAP for autocomplete: {e}")
        return {"suggestions": []}
    
    # Full search for longer queries
    cols = "pl_name,pl_rade,pl_masse,st_rad,st_lum,pl_orbsmax,ra,dec"
    
    # Simple search on exact pl_name or exact hostname, or like prefix
    adql_queries = [
        f"SELECT {cols} FROM pscomppars WHERE LOWER(pl_name)=LOWER('{query_str}')",
        f"SELECT {cols} FROM pscomppars WHERE LOWER(hostname)=LOWER('{query_str}')",
        f"SELECT {cols} FROM pscomppars WHERE LOWER(pl_name) LIKE LOWER('{query_str}%')",
        f"SELECT {cols} FROM pscomppars WHERE LOWER(hostname) LIKE LOWER('{query_str}%')"
    ]
    
    async with httpx.AsyncClient() as client:
        for adql in adql_queries:
            try:
                response = await client.get(
                    NASA_TAP_URL,
                    params={"query": adql, "format": "json"},
                    timeout=15.0
                )
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        # Find the row with most non-null values
                        def count_non_null(row):
                            return sum(1 for k in row.keys() if row[k] is not None)
                        best_row = max(data, key=count_non_null)
                        
                        # Convert st_lum (log) to solar luminosity if present
                        st_lum = None
                        if best_row.get("st_lum") is not None:
                            st_lum = 10 ** float(best_row["st_lum"])
                            
                        return {
                            "pl_name": best_row.get("pl_name"),
                            "pl_rade": best_row.get("pl_rade"),
                            "pl_masse": best_row.get("pl_masse"),
                            "st_rad": best_row.get("st_rad"),
                            "st_lum": st_lum,
                            "pl_orbsmax": best_row.get("pl_orbsmax"),
                            "ra": best_row.get("ra"),
                            "dec": best_row.get("dec")
                        }
            except Exception as e:
                print(f"Error querying NASA TAP: {e}")
                
    raise HTTPException(status_code=404, detail="Planet not found in NASA Exoplanet Archive")

@app.get("/api/autocomplete")
async def autocomplete_planets(q: str = Query(..., description="Partial planet or star name")):
    """
    Return up to 10 planet names that match the query prefix.
    """
    query_str = q.strip().replace("'", "''").lower()
    if len(query_str) < 2:
        return {"suggestions": []}
    
    adql = f"SELECT DISTINCT pl_name FROM pscomppars WHERE LOWER(pl_name) LIKE LOWER('{query_str}%') LIMIT 10"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                NASA_TAP_URL,
                params={"query": adql, "format": "json"},
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                suggestions = [row["pl_name"] for row in data if row["pl_name"]]
                return {"suggestions": suggestions}
        except Exception as e:
            print(f"Error querying NASA TAP for autocomplete: {e}")
    
    return {"suggestions": []}

@app.get("/api/recommendations")
async def get_recommendations():
    """
    Return 20 famous exoplanets with descriptions.
    """
    return {"recommendations": FAMOUS_EXOPLANETS}

@app.get("/api/planet/{name}/images")
async def get_planet_images(name: str):
    """
    Search NASA Images API for photos/videos of that planet.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                NASA_IMAGES_API_URL,
                params={"q": name, "media_type": "image"},
                timeout=15.0
            )
            if response.status_code == 200:
                data = response.json()
                items = data.get("collection", {}).get("items", [])
                
                results = []
                # Return up to 5 images
                for item in items[:5]:
                    links = item.get("links", [])
                    data_info = item.get("data", [{}])[0]
                    if links:
                        results.append({
                            "title": data_info.get("title", ""),
                            "description": data_info.get("description", ""),
                            "image_url": links[0].get("href", "")
                        })
                return {"images": results}
            else:
                return {"images": []}
        except Exception as e:
            print(f"Error querying NASA Images API: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch images from NASA")

@app.get("/api/planet/{name}/details")
async def get_planet_details(name: str):
    """
    Full analysis: composition, habitability, atmosphere, habitable zone calculation.
    """
    # First, fetch the planet's data from our own search logic
    try:
        planet_data = await search_planet(q=name)
    except HTTPException:
        raise HTTPException(status_code=404, detail="Planet data not found for analysis")

    st_lum = planet_data.get("st_lum")
    pl_orbsmax = planet_data.get("pl_orbsmax")
    pl_masse = planet_data.get("pl_masse")
    pl_rade = planet_data.get("pl_rade")

    # 1. Habitable Zone Calculation
    # Conservative Habitable Zone (Kopparapu et al. 2013 approximations)
    # Inner = sqrt(L_star / 1.1)
    # Outer = sqrt(L_star / 0.53)
    hz_inner = None
    hz_outer = None
    is_habitable = False
    
    if st_lum is not None and st_lum > 0:
        hz_inner = math.sqrt(st_lum / 1.1)
        hz_outer = math.sqrt(st_lum / 0.53)
        
        if pl_orbsmax is not None:
            if hz_inner <= pl_orbsmax <= hz_outer:
                is_habitable = True

    # 2. Composition Calculation based on Density
    # Earth Density ~ 5.51 g/cm³
    # Earth Mass = 5.972 × 10^24 kg
    # Earth Radius = 6371 km
    composition = "Unknown"
    planet_type = "Unknown"
    density_g_cm3 = None
    has_atmosphere = "Unknown"

    if pl_masse is not None and pl_rade is not None and pl_rade > 0:
        # Volume relative to Earth (V = (4/3) * pi * R^3)
        # So Volume_ratio = pl_rade^3
        volume_ratio = pl_rade ** 3
        # Density ratio = Mass_ratio / Volume_ratio
        density_ratio = pl_masse / volume_ratio
        density_g_cm3 = density_ratio * 5.51

        if density_g_cm3 < 2.0:
            composition = "Primarily Gas/Ice (Hydrogen, Helium, Water)"
            has_atmosphere = "Yes (Thick gas envelope)"
            if pl_masse > 10:
                planet_type = "Gas Giant / Ice Giant"
            else:
                planet_type = "Mini-Neptune / Water World"
        elif 2.0 <= density_g_cm3 < 4.0:
            composition = "Mixed (Rock, Ice, Gas)"
            has_atmosphere = "Likely (Significant envelope)"
            planet_type = "Super-Earth / Ocean World"
        elif 4.0 <= density_g_cm3 < 8.0:
            composition = "Primarily Rocky (Silicates, Iron)"
            has_atmosphere = "Possible (Thin to moderate atmosphere)"
            if pl_rade <= 1.5:
                planet_type = "Terrestrial / Earth-like"
            else:
                planet_type = "Rocky Super-Earth"
        else:
            composition = "Iron-Rich / Extremely Dense Rocky"
            has_atmosphere = "Unlikely (Too dense/hot, atmosphere possibly stripped)"
            planet_type = "Iron Planet / Mega-Earth"

    return {
        "planet": planet_data.get("pl_name"),
        "analysis": {
            "is_in_habitable_zone": is_habitable,
            "habitable_zone_inner_au": round(hz_inner, 3) if hz_inner else None,
            "habitable_zone_outer_au": round(hz_outer, 3) if hz_outer else None,
            "planet_type": planet_type,
            "composition_estimate": composition,
            "density_g_cm3": round(density_g_cm3, 2) if density_g_cm3 else None,
            "atmosphere_prediction": has_atmosphere
        },
        "raw_data": planet_data
    }
