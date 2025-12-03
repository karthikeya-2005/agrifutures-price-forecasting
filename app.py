import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
from datetime import timedelta, datetime
from enhanced_predictor import predict_with_forecast, fetch_current_market_conditions, predict_price
from geolocation_fetcher import get_coordinates
# Note: We now use only training data, not locations_data for UI
# from locations_data import INDIA_STATES, INDIA_DISTRICTS, INDIA_CROPS  # Kept for reference only
from get_available_commodities import (
    get_available_commodities, 
    get_location_info,
    get_available_districts,
    get_available_states,
    is_valid_combination
)
from location_normalizer import normalize_state_name, normalize_district_name, normalize_location, get_all_states, get_all_districts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Agricultural Price Prediction - India",
    page_icon="🌾",
    layout="wide"
)

# Minimalist Dark Mode with Gradients
st.markdown("""
    <style>
    /* Minimalist Dark Mode Color Palette */
    :root {
        --primary: #6366F1;
        --primary-dark: #4F46E5;
        --primary-light: #818CF8;
        --secondary: #10B981;
        --accent: #F59E0B;
        --error: #EF4444;
        --warning: #F59E0B;
        --success: #10B981;
        --info: #3B82F6;
        --background: #0F172A;
        --surface: #1E293B;
        --surface-light: #334155;
        --text-primary: #F1F5F9;
        --text-secondary: #CBD5E1;
        --border: #334155;
        --shadow: rgba(0, 0, 0, 0.3);
    }
    
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    }
    
    /* Header Styling - Minimalist Dark */
    h1 {
        color: #F1F5F9;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 2rem;
        background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #F1F5F9;
        font-weight: 500;
        border-left: 3px solid #6366F1;
        padding-left: 1rem;
        margin-top: 2rem;
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #CBD5E1;
        font-weight: 500;
        font-size: 1.1rem;
        margin-top: 1.5rem;
    }
    
    /* Metric Cards - Minimalist Dark */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #F1F5F9;
        letter-spacing: -0.5px;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Metric Container - Dark Gradient Cards */
    div[data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetricContainer"]:hover {
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.4);
        border-color: #6366F1;
        transform: translateY(-2px);
        background: linear-gradient(135deg, #334155 0%, #1E293B 100%);
    }
    
    /* Sidebar Styling - Dark */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border-right: 1px solid #334155;
    }
    
    /* Button Styling - Minimalist Dark Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%);
        color: #FFFFFF;
        font-weight: 500;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.3);
    }
    
    /* Success/Warning/Error Messages - Dark Gradient */
    .stSuccess {
        background: linear-gradient(135deg, #1E293B 0%, #064E3B 100%);
        color: #10B981;
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #10B981;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #1E293B 0%, #78350F 100%);
        color: #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #F59E0B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .stError {
        background: linear-gradient(135deg, #1E293B 0%, #7F1D1D 100%);
        color: #EF4444;
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #EF4444;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1E293B 0%, #1E3A8A 100%);
        color: #3B82F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Dataframe Styling - Dark */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #334155;
        background: #1E293B;
    }
    
    .dataframe thead {
        background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%);
        color: #FFFFFF;
        font-weight: 500;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background: #334155;
    }
    
    .dataframe tbody tr {
        color: #F1F5F9;
    }
    
    .dataframe tbody tr:hover {
        background: #475569;
    }
    
    /* Selectbox Styling - Dark */
    .stSelectbox label {
        font-weight: 500;
        color: #CBD5E1;
        font-size: 0.875rem;
    }
    
    .stSelectbox > div > div {
        background: #1E293B;
        border: 1px solid #334155;
        color: #F1F5F9;
    }
    
    /* Spinner - Dark */
    .stSpinner > div {
        border-color: #334155;
        border-top-color: #6366F1;
    }
    
    /* Card-like containers - Dark Gradient */
    .element-container {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    
    .element-container:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        border-color: #475569;
    }
    
    /* Price highlight - Gradient */
    .price-highlight {
        background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%);
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        color: #FFFFFF;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.3);
    }
    
    /* Text colors for dark mode */
    p, span, div {
        color: #CBD5E1;
    }
    
    /* Input styling */
    input, textarea {
        background: #1E293B !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("Agricultural Commodity Price Prediction - India")
    st.markdown("### Advanced Prediction with Weather & Market Conditions")
    
    # Initialize session state for tracking state/district changes
    # Clear any old cached data to ensure fresh load from consolidated data
    if 'prev_state' not in st.session_state:
        st.session_state.prev_state = None
    if 'prev_district' not in st.session_state:
        st.session_state.prev_district = None
    if 'available_crops' not in st.session_state:
        st.session_state.available_crops = []
    if 'location_info' not in st.session_state:
        st.session_state.location_info = {}
    if 'available_crops' not in st.session_state:
        st.session_state.available_crops = []
    
    # Force reload of mappings to ensure fresh data from consolidated.csv
    from get_available_commodities import reload_mappings
    reload_mappings()
    
    # Sidebar
    with st.sidebar:
        st.header("Input Parameters")
        
        # Get available states from training data ONLY (no fallback)
        available_states = get_available_states()
        
        if not available_states:
            st.error("❌ No training data available. Please ensure consolidated data file exists.")
            st.info("💡 Expected file: `data/combined/all_sources_consolidated.csv`")
            st.stop()
        
        state = st.selectbox(
            "Select State", 
            available_states,
            help=f"{len(available_states)} states available in training data"
        )
        
        # Get available districts for selected state from training data ONLY
        available_districts = get_available_districts(state)
        
        if not available_districts:
            st.warning(f"⚠️ No districts found in training data for {state}.")
            st.info("💡 This state may not have sufficient training data.")
            district = None
        else:
            district = st.selectbox(
                "Select District", 
                available_districts,
                help=f"{len(available_districts)} districts available in {state}"
            )
        
        # Get available commodities for selected state and district
        if state != st.session_state.prev_state or district != st.session_state.prev_district:
            # State or district changed, update available crops
            if district:
                st.session_state.available_crops = get_available_commodities(state, district)
                st.session_state.location_info = get_location_info(state, district)
            else:
                st.session_state.available_crops = []
                st.session_state.location_info = {}
            st.session_state.prev_state = state
            st.session_state.prev_district = district
        
        available_crops = st.session_state.available_crops if district else []
        location_info = st.session_state.get('location_info', {})
        
        # Initialize on first run
        if district and not available_crops and (st.session_state.prev_state is None or st.session_state.prev_district is None):
            available_crops = get_available_commodities(state, district)
            location_info = get_location_info(state, district)
            st.session_state.available_crops = available_crops
            st.session_state.location_info = location_info
            st.session_state.prev_state = state
            st.session_state.prev_district = district
        
        # Show crop selection
        if district and available_crops:
            crop = st.selectbox(
                "Select Crop/Commodity", 
                available_crops,
                help=f"{len(available_crops)} commodities available for {district}, {state} (based on training data)"
            )
            
            # Show detailed info
            if location_info.get('has_location_specific'):
                st.success(
                    f"[OK] {len(available_crops)} commodities available for {district}, {state}"
                )
            else:
                st.success(f"[OK] {len(available_crops)} commodities available (based on training data)")
        elif district:
            st.warning(f"No commodities found in training data for {district}, {state}")
            st.info("💡 Please select a different location that has training data.")
            crop = None
        else:
            crop = None
        
        target_date = st.date_input("Target Date", value=datetime.now().date() + timedelta(days=7))
        forecast_days = st.slider("Forecast Period (days)", 1, 7, 7)
        
        st.markdown("---")
        st.markdown("### Model Information")
        if available_crops:
            st.info(f"Using pre-trained models for {len(available_crops)} commodities:\n- Weather data integration\n- Current market conditions\n- Historical patterns\n- Location-specific predictions")
        else:
            st.info("Using pre-trained advanced models with:\n- Weather data integration\n- Current market conditions\n- Historical patterns")
    
    # Main content
    if crop is None:
        if not district:
            st.info("Please select a state and district to see available commodities.")
        else:
            st.info("Please select a state, district, and crop to generate predictions.")
        return
    
    # Try to normalize state and district, but use original if normalization fails
    # This allows the app to work even if location_normalizer doesn't have all locations
    normalized_state, normalized_district = normalize_location(state, district)
    
    # Use normalized names if available, otherwise use original names
    # The data matching functions handle both cases
    if normalized_state:
        state_for_prediction = normalized_state
    else:
        state_for_prediction = state
    
    if normalized_district:
        district_for_prediction = normalized_district
    else:
        district_for_prediction = district
    
    # Validate combination before prediction
    # Since crop came from dropdown (which only shows valid crops), validation should pass
    # We check silently - no need to show warnings since dropdown already ensures validity
    # This validation is just for internal consistency checks
    is_valid = is_valid_combination(state, district, crop)
    if not is_valid:
        # Try with normalized names as fallback
        is_valid = is_valid_combination(state_for_prediction, district_for_prediction, crop)
    
    # Silently proceed - dropdown ensures only valid combinations are shown
    # No need to warn user since they can only select valid combinations
    
    if st.button("Generate Prediction & Forecast", type="primary"):
        # Create progress tracking
        progress_container = st.container()
        status_text = progress_container.empty()
        progress_bar = progress_container.progress(0)
        
        def update_progress(status, step):
            """Update progress indicator"""
            step_mapping = {
                'init': 0.05,
                'market_data': 0.30,
                'coordinates': 0.35,
                'weather': 0.50,
                'model': 0.60,
                'prediction': 0.90,
                'complete': 1.0
            }
            progress = step_mapping.get(step, 0.5)
            progress_bar.progress(progress)
            status_text.text(status)
        
        try:
            # Validate inputs before prediction
            if not state_for_prediction or not district_for_prediction or not crop:
                st.error("❌ Please select state, district, and crop before generating prediction.")
                return
            
            if target_date < datetime.now().date():
                st.warning("⚠️ Target date is in the past. Using today's date instead.")
                target_date = datetime.now().date()
            
            # Ensure forecast_days is within valid range (1-7)
            forecast_days = max(1, min(7, forecast_days))
            
            # Use normalized names for prediction if available, otherwise use original
            # The predictor handles both cases
            update_progress("Starting prediction...", "init")
            forecast_result = predict_with_forecast(
                state_for_prediction, 
                district_for_prediction, 
                crop, 
                target_date, 
                days_ahead=forecast_days,
                progress_callback=update_progress
            )
            update_progress("Prediction complete!", "complete")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Validate result
            if forecast_result is None:
                st.error("❌ Prediction failed. Please try again or check if model exists for this combination.")
                st.info("💡 Tip: Ensure the selected state, district, and crop combination has training data.")
                return
                    
        except ValueError as e:
            st.error(f"❌ Invalid input: {str(e)}")
            st.info("💡 Please check your selections and try again.")
            forecast_result = None
        except KeyError as e:
            st.error(f"❌ Missing required data: {str(e)}")
            st.info("💡 The system may be missing some required data files. Please check data files exist.")
            forecast_result = None
        except Exception as e:
            st.error(f"❌ Error generating prediction: {str(e)}")
            st.info("💡 Tip: If weather data fails, the system will use default values. This is normal if the weather API rate limit is reached.")
            import traceback
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
            forecast_result = None
        
        # Display results if prediction was successful
        if forecast_result:
            predictions = forecast_result.get('predictions', [])
            market_conditions = forecast_result.get('market_conditions', {})
            
            # Check for errors
            if forecast_result.get('error'):
                st.error(f"⚠️ {forecast_result['error']}")
                st.info("💡 The system will try to use available data sources. Some sources may be temporarily unavailable.")
            
            # Debug: Check predictions structure
            if not predictions or len(predictions) == 0:
                st.warning("⚠️ No predictions generated. This may be due to:")
                st.info("""
                - Model not found for this combination
                - Weather data unavailable
                - Market data sources temporarily unavailable
                
                💡 **The system is trying multiple data sources:**
                - e-NAM (trade-data, Agm_Enam_ctrl, live_price)
                - Commodity Online
                - NCDEX Spot Prices
                - Data.gov.in
                - AGMARKNET
                """)
                
                # Show what data was found
                if market_conditions:
                    st.info(f"✅ Market conditions found: Current price ₹{market_conditions.get('current_price', 'N/A')}")
                else:
                    st.info("❌ No market conditions found from any source")
                
                return
            
            # Current market conditions with data source info
            st.header("📊 Current Market Conditions")
            
            # Check if market data is available
            has_market_data = market_conditions and market_conditions.get('current_price') is not None
            
            # Check if using fallback data
            is_fallback = market_conditions and market_conditions.get('is_fallback', False)
            
            if not has_market_data:
                st.warning("⚠️ **Current data unavailable** - Market data for this commodity and location is currently unavailable from all data sources.")
                st.info("💡 **Note:** The forecast graph below will still display predictions based on historical patterns, weather data, and model analysis. The predictions may be less accurate without current market data.")
            else:
                # Show data source if available
                if market_conditions.get('data_source'):
                    data_source = market_conditions['data_source']
                    source_labels = {
                        'enam_primary': 'e-NAM (PRIMARY Source)',
                        'location_specific': 'Location-Specific Data',
                        'state_level': 'State-Level Data',
                        'commodity_level': 'Commodity-Level Data',
                        'previous_week_fallback': 'Previous Week Data (Fallback)'
                    }
                    source_colors = {
                        'enam_primary': 'success',
                        'location_specific': 'success',
                        'state_level': 'info',
                        'commodity_level': 'warning',
                        'previous_week_fallback': 'warning'
                    }
                    
                    source_label = source_labels.get(data_source, data_source)
                    source_color = source_colors.get(data_source, 'info')
                    
                    if is_fallback:
                        st.warning(f"⚠️ **Using Previous Week's Data (Fallback)**")
                        if market_conditions.get('fallback_date_range'):
                            st.info(f"📅 **Date Range:** {market_conditions['fallback_date_range']}")
                        st.info(f"💡 Current data unavailable. Using data from previous week for predictions. ({market_conditions.get('records_count', 0)} records)")
                    else:
                        if data_source == 'enam_primary':
                            st.success(f"✅ **PRIMARY SOURCE: e-NAM** ({market_conditions.get('records_count', 0)} records)")
                        else:
                            st.info(f"📡 Data Source: {source_label} ({market_conditions.get('records_count', 0)} records)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Get price unit from forecast result
            price_unit = forecast_result.get('price_unit_display', '₹ per quintal')
            quantity_unit = forecast_result.get('quantity_unit', 'quintal')
            
            with col1:
                if has_market_data:
                    st.metric("Current Price", f"₹{market_conditions['current_price']:,.2f}", help=f"Price per {quantity_unit}")
                else:
                    st.metric("Current Price", "N/A", help=f"Price per {quantity_unit}")
            
            with col2:
                if market_conditions and market_conditions.get('avg_price_7d'):
                    st.metric("7-Day Average", f"₹{market_conditions['avg_price_7d']:,.2f}", help=f"Average price per {quantity_unit} over last 7 days")
                else:
                    st.metric("7-Day Average", "N/A", help=f"Average price per {quantity_unit} over last 7 days")
            
            with col3:
                if market_conditions and market_conditions.get('price_trend'):
                    trend = market_conditions['price_trend']
                    trend_icon = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"
                    trend_color = "#4CAF50" if trend == 'increasing' else "#F44336" if trend == 'decreasing' else "#FF9800"
                    st.markdown(f"<div style='text-align: center;'><div style='font-size: 1.5rem; color: {trend_color};'>{trend_icon}</div><div style='font-weight: 600;'>{trend.title()}</div></div>", unsafe_allow_html=True)
                else:
                    st.metric("Price Trend", "N/A")
            
            with col4:
                if market_conditions and market_conditions.get('volatility'):
                    st.metric("Volatility", f"₹{market_conditions['volatility']:,.2f}", help=f"Price volatility (standard deviation) per {quantity_unit}")
                else:
                    st.metric("Volatility", "N/A", help=f"Price volatility per {quantity_unit}")
            
            # Main prediction
            st.header("Price Prediction")
            
            # Show current price prominently alongside prediction
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                if has_market_data:
                    st.metric(
                        "Current Price",
                        f"₹{market_conditions['current_price']:,.2f}",
                        help=f"Latest available market price per {quantity_unit} (1 {quantity_unit} = 100 kg)"
                    )
                else:
                    st.metric("Current Price", "N/A", help=f"Current market data unavailable (price per {quantity_unit})")
            
            # Find target prediction - handle both date and datetime.date types
            target_prediction = None
            for p in predictions:
                pred_date = p.get('date')
                # Convert both to date objects for comparison
                if hasattr(pred_date, 'date'):
                    pred_date = pred_date.date()
                if hasattr(target_date, 'date'):
                    target_date_compare = target_date.date()
                else:
                    target_date_compare = target_date
                
                if pred_date == target_date_compare:
                    target_prediction = p
                    break
            
            if target_prediction:
                with pred_col2:
                    st.metric(
                        f"Predicted Price ({target_date})",
                        f"₹{target_prediction.get('price', 0):,.2f}",
                        help=f"Forecasted price per {quantity_unit} for {target_date} (1 {quantity_unit} = 100 kg)"
                    )
                
                with pred_col3:
                    if has_market_data:
                        try:
                            target_price = float(target_prediction.get('price', 0))
                            current_price_val = float(market_conditions['current_price'])
                            
                            if target_price > 0 and current_price_val > 0:
                                change = target_price - current_price_val
                                change_pct = (change / current_price_val) * 100
                                st.metric(
                                    "Expected Change",
                                    f"₹{change:+,.2f}",
                                    f"{change_pct:+.2f}%",
                                    help="Difference between predicted and current price"
                                )
                            else:
                                st.metric("Expected Change", "N/A")
                        except (ValueError, TypeError, KeyError) as e:
                            st.metric("Expected Change", "N/A")
                    else:
                        st.metric("Expected Change", "N/A", help="Cannot calculate change without current market data")
            
            # Forecast Graph
            st.header("Price Forecast Graph")
            
            try:
                # Validate predictions data
                if not predictions or len(predictions) == 0:
                    st.warning("No forecast data available to display.")
                    return
                
                # Convert predictions to DataFrame
                forecast_df = pd.DataFrame(predictions)
                
                # Ensure date column exists and is properly formatted
                if 'date' not in forecast_df.columns:
                    st.error(f"Forecast data missing date column. Available columns: {list(forecast_df.columns)}")
                    return
                
                # Convert date column
                forecast_df['date'] = pd.to_datetime(forecast_df['date'], errors='coerce')
                
                # Remove any rows with invalid dates
                forecast_df = forecast_df.dropna(subset=['date'])
                
                if len(forecast_df) == 0:
                    st.warning("No valid forecast data after processing dates.")
                    return
                
                # Ensure price column exists
                if 'price' not in forecast_df.columns:
                    st.error(f"Forecast data missing price column. Available columns: {list(forecast_df.columns)}")
                    return
                
                # Ensure price values are numeric
                forecast_df['price'] = pd.to_numeric(forecast_df['price'], errors='coerce')
                forecast_df = forecast_df.dropna(subset=['price'])
                
                if len(forecast_df) == 0:
                    st.warning("No valid forecast data after processing prices.")
                    return
                
                # Sort by date
                forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
                
                fig = go.Figure()
                
                # Add historical prices first if available (for fill reference)
                has_historical = False
                if market_conditions and market_conditions.get('recent_prices'):
                    try:
                        historical = pd.DataFrame(market_conditions['recent_prices'])
                        if 'date' in historical.columns and 'price' in historical.columns:
                            historical['date'] = pd.to_datetime(historical['date'], errors='coerce')
                            historical = historical.dropna(subset=['date', 'price'])
                            if len(historical) > 0:
                                historical = historical.sort_values('date')
                                fig.add_trace(go.Scatter(
                                    x=historical['date'],
                                    y=historical['price'],
                                    mode='lines+markers',
                                    name='Historical Prices',
                                    line=dict(color='#10B981', width=2, dash='dot'),
                                    marker=dict(size=5, color='#10B981', opacity=0.7, line=dict(width=1, color='#1E293B')),
                                    opacity=0.7,
                                    showlegend=True
                                ))
                                has_historical = True
                    except Exception as e:
                        logger.warning(f"Could not add historical data: {str(e)}")
                        # Continue without historical data - graph will still render
                
                # Forecast line - Dark mode colors with gradient fill
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['price'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#6366F1', width=3, shape='spline'),
                    marker=dict(size=8, color='#818CF8', line=dict(width=1, color='#1E293B')),
                    fill='tonexty' if has_historical else 'tozeroy',
                    fillcolor='rgba(99, 102, 241, 0.2)'
                ))
            
                # Current price - Dark mode colors (only if market data is available)
                if has_market_data:
                    try:
                        current_date = datetime.now().date()
                        current_price = float(market_conditions['current_price'])
                        if not pd.isna(current_price) and current_price > 0:
                            fig.add_trace(go.Scatter(
                                x=[pd.Timestamp(current_date)],
                                y=[current_price],
                                mode='markers+text',
                                name='Current Price',
                                text=[f'Current: ₹{current_price:,.0f}'],
                                textposition='top center',
                                textfont=dict(size=12, color='#F59E0B', family='Arial', weight='bold'),
                                marker=dict(
                                    size=20, 
                                    color='#F59E0B', 
                                    symbol='diamond',
                                    line=dict(width=2, color='#1E293B'),
                                    opacity=1.0
                                ),
                                hovertemplate='<b style="color:#F59E0B;">Current Price</b><br>Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
                            ))
                    except Exception as e:
                        logger.warning(f"Could not add current price marker: {str(e)}")
                
                # Highlight target date
                if target_prediction:
                    try:
                        target_price = float(target_prediction.get('price', 0))
                        if target_price > 0:
                                # Get target date from prediction or use target_date
                                pred_date = target_prediction.get('date')
                                if pred_date:
                                    # Convert to datetime for Plotly
                                    if isinstance(pred_date, str):
                                        target_date_plotly = pd.to_datetime(pred_date)
                                    elif hasattr(pred_date, 'to_pydatetime'):
                                        target_date_plotly = pred_date.to_pydatetime()
                                    elif hasattr(pred_date, 'date'):
                                        target_date_plotly = pd.Timestamp(pred_date.date())
                                    else:
                                        target_date_plotly = pd.Timestamp(pred_date)
                                    
                                    # Find the corresponding price in forecast_df
                                    forecast_price = None
                                    for idx, row in forecast_df.iterrows():
                                        row_date = row['date']
                                        if isinstance(row_date, pd.Timestamp):
                                            row_date = row_date.date()
                                        elif hasattr(row_date, 'date'):
                                            row_date = row_date.date()
                                        
                                        pred_date_compare = pred_date
                                        if hasattr(pred_date_compare, 'date'):
                                            pred_date_compare = pred_date_compare.date()
                                        
                                        if row_date == pred_date_compare:
                                            forecast_price = row['price']
                                            break
                                    
                                    if forecast_price is None:
                                        forecast_price = target_price
                                    
                                    # Add vertical line
                                    fig.add_vline(
                                        x=target_date_plotly,
                                        line_dash="dash",
                                        line_color="#F59E0B",
                                        line_width=2,
                                        annotation_text=f"Target: ₹{target_price:,.2f}",
                                        annotation_position="top",
                                        annotation_font_color="#F59E0B",
                                        annotation_bgcolor="rgba(30, 41, 59, 0.8)",
                                        annotation_bordercolor="#F59E0B"
                                    )
                    except Exception as e:
                        # Fallback: add annotation without vline if there's an error
                        try:
                            target_price = float(target_prediction.get('price', 0))
                            if target_price > 0:
                                pred_date = target_prediction.get('date')
                                if pred_date:
                                    if isinstance(pred_date, str):
                                        target_date_plotly = pd.to_datetime(pred_date)
                                    elif hasattr(pred_date, 'to_pydatetime'):
                                        target_date_plotly = pred_date.to_pydatetime()
                                    elif hasattr(pred_date, 'date'):
                                        target_date_plotly = pd.Timestamp(pred_date.date())
                                    else:
                                        target_date_plotly = pd.Timestamp(pred_date)
                                    
                                    fig.add_annotation(
                                        x=target_date_plotly,
                                        y=target_price,
                                        text=f"Target: ₹{target_price:,.2f}",
                                        showarrow=True,
                                        arrowhead=2,
                                        bgcolor="rgba(245, 158, 11, 0.8)",
                                        bordercolor="#F59E0B",
                                        font=dict(color="#F1F5F9", size=11)
                                    )
                        except Exception as annot_error:
                            # Silently skip if annotation also fails
                            pass
                
                # Get price unit for graph title
                price_unit_display = forecast_result.get('price_unit_display', '₹ per quintal')
                quantity_info = f" (1 {quantity_unit} = 100 kg)"
                
                fig.update_layout(
                    title=dict(
                        text=f"{crop} Price Forecast - {district}, {state}{quantity_info}",
                        font=dict(size=18, color='#F1F5F9', family='Arial', weight='bold'),
                        x=0.5,
                        xanchor='center',
                        pad=dict(b=20)
                    ),
                    xaxis=dict(
                        title=dict(text="Date", font=dict(size=12, color='#CBD5E1', weight='bold')),
                        gridcolor='#334155',
                        showgrid=True,
                        gridwidth=1,
                        linecolor='#475569',
                        linewidth=1,
                        zeroline=False
                    ),
                    yaxis=dict(
                        title=dict(text="Price (₹ per quintal)", font=dict(size=12, color='#CBD5E1', weight='bold')),
                        gridcolor='#334155',
                        showgrid=True,
                        gridwidth=1,
                        linecolor='#475569',
                        linewidth=1,
                        zeroline=False
                    ),
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(30, 41, 59, 0.95)',
                        bordercolor='#475569',
                        borderwidth=1,
                        font=dict(size=11, color='#CBD5E1', weight='normal'),
                        x=0.02,
                        y=0.98,
                        xanchor='left',
                        yanchor='top'
                    ),
                    plot_bgcolor='#1E293B',
                    paper_bgcolor='#1E293B',
                    font=dict(family='Arial', size=11, color='#CBD5E1', weight='normal'),
                    margin=dict(l=60, r=20, t=60, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                    st.error(f"Error creating forecast graph: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    # Show raw data for debugging
                    if predictions:
                        st.write("Raw predictions data:", predictions[:5])
            
            # Forecast table
            st.header("Detailed Forecast")
            
            # Display quantity/unit information
            st.info(f"📦 **Price Unit:** {price_unit} (1 {quantity_unit} = 100 kg)")
            
            # Add current price as first row if available
            try:
                display_df = forecast_df.copy()
                # Ensure date column is datetime before formatting
                if not pd.api.types.is_datetime64_any_dtype(display_df['date']):
                    display_df['date'] = pd.to_datetime(display_df['date'], errors='coerce')
                display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df[f'Price ({price_unit})'] = display_df['price'].apply(lambda x: f"₹{float(x):,.2f}" if pd.notna(x) else "N/A")
            except Exception as e:
                st.error(f"Error preparing forecast table: {str(e)}")
                return
            
            if has_market_data:
                try:
                    current_price = float(market_conditions['current_price'])
                    current_date = datetime.now().date().strftime('%Y-%m-%d')
                    
                    # Create current price row with correct column name
                    price_col_name = f'Price ({price_unit})'
                    current_row = pd.DataFrame([{
                        'Date': f"{current_date} (Current)",
                        price_col_name: f"₹{current_price:,.2f}",
                        'Change from Current': "₹0.00",
                        'Change %': "0.00%"
                    }])
                    
                    # Add change columns to forecast dataframe
                    display_df['Change from Current'] = (display_df['price'] - current_price).apply(
                        lambda x: f"₹{float(x):+,.2f}" if pd.notna(x) else "N/A"
                    )
                    display_df['Change %'] = ((display_df['price'] - current_price) / current_price * 100).apply(
                        lambda x: f"{float(x):+.2f}%" if pd.notna(x) else "N/A"
                    )
                    
                    # Combine current price row with forecast
                    display_df = pd.concat([current_row, display_df], ignore_index=True)
                    
                    cols_to_show = ['Date', price_col_name, 'Change from Current', 'Change %']
                except Exception as e:
                    logger.warning(f"Could not add current price comparison: {str(e)}")
                    price_col_name = f'Price ({price_unit})'
                    cols_to_show = ['Date', price_col_name]
            else:
                price_col_name = f'Price ({price_unit})'
                cols_to_show = ['Date', price_col_name]
            
            # Highlight current price row
            st.dataframe(
                display_df[cols_to_show],
                use_container_width=True,
                hide_index=True
            )
            
            # Add note about current price and quantity
            if has_market_data:
                st.info(f"💡 **Current market price:** ₹{market_conditions['current_price']:,.2f} {price_unit} (as of {datetime.now().date()}) | **Quantity:** 1 {quantity_unit} = 100 kg")
            else:
                st.info(f"💡 **Forecast based on:** Historical patterns, weather data, and model analysis. Current market data unavailable. | **Price Unit:** {price_unit} (1 {quantity_unit} = 100 kg)")
            
            # Weather information
            if predictions and len(predictions) > 0 and predictions[0].get('weather'):
                st.header("🌤️ Weather Conditions")
                weather = predictions[0]['weather']
                if weather:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        temp_max = weather.get('temperature_2m_max', 'N/A')
                        st.metric("Max Temperature", f"{temp_max}°C" if temp_max != 'N/A' else "N/A")
                    with col2:
                        temp_min = weather.get('temperature_2m_min', 'N/A')
                        st.metric("Min Temperature", f"{temp_min}°C" if temp_min != 'N/A' else "N/A")
                with col3:
                    st.metric("Precipitation", f"{weather.get('precipitation_sum', 'N/A')} mm")
            
            # Factors considered
            st.header("ℹ️ Factors Considered in Prediction")
            factors_col1, factors_col2 = st.columns(2)
            
            with factors_col1:
                st.markdown("""
                **✅ Weather Conditions:**
                - Temperature (max/min)
                - Precipitation
                - Seasonal patterns
                
                **✅ Current Market Conditions:**
                - Recent prices
                - Price trends
                - Market volatility
                """)
            
            with factors_col2:
                st.markdown("""
                **✅ Historical Patterns:**
                - Seasonal variations
                - Year-over-year trends
                - Location-specific patterns
                
                **✅ Advanced Features:**
                - Price momentum
                - Moving averages
                - Temporal features
                """)
        
        else:
            st.error("Could not generate prediction. Please check:")
            st.markdown("""
            - Commodity name matches available models
            - Location coordinates are available
            - Model exists for this commodity
            """)
            
            st.info("💡 Tip: The system has pre-trained models for 69 commodities. Make sure the commodity name matches exactly.")

if __name__ == "__main__":
    main()
