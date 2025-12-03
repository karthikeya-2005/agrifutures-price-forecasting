import requests
from location_normalizer import normalize_state_name, normalize_district_name
from cache_manager import get_cache_manager

def get_coordinates(state, district):
    """
    Get latitude and longitude for given state and district using Nominatim API.
    Uses caching to avoid repeated API calls for the same location.
    :param state: string (will be normalized)
    :param district: string (will be normalized)
    :return: tuple (latitude, longitude) or (None, None) if not found
    """
    # Check cache first
    cache_manager = get_cache_manager()
    cached_result = cache_manager.get('coordinates', state, district)
    if cached_result is not None:
        return cached_result
    
    # Normalize state and district names
    normalized_state = normalize_state_name(state)
    normalized_district = normalize_district_name(district, normalized_state)
    
    # Use normalized names for geocoding
    state_to_use = normalized_state if normalized_state else state
    district_to_use = normalized_district if normalized_district else district
    
    base_url = "https://nominatim.openstreetmap.org/search"
    query = f"{district_to_use}, {state_to_use}, India"
    params = {
        "q": query,
        "format": "json",
        "limit": 1
    }
    try:
        response = requests.get(base_url, params=params, timeout=10, headers={"User-Agent": "AgrifuturesApp/1.0"})
        response.raise_for_status()
        data = response.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            result = (lat, lon)
            # Cache the result
            cache_manager.set('coordinates', result, state=state, district=district)
            return result
        else:
            result = (None, None)
            # Cache None for shorter time
            cache_manager.set('coordinates', result, ttl=3600, state=state, district=district)
            return result
    except Exception as e:
        print(f"Failed to get coordinates: {e}")
        result = (None, None)
        # Cache None for shorter time
        cache_manager.set('coordinates', result, ttl=3600, state=state, district=district)
        return result
