# File Documentation - Complete Reference Guide

This document provides a comprehensive explanation of each file in the project, including its purpose, functionality, and usage context.

**Last Updated**: December 2024

---

## 📱 Production Application Files

### `app.py`
**Purpose**: Main Streamlit web application for agricultural price prediction

**Functionality**:
- Interactive web interface for price predictions
- State/District/Crop selection with automatic filtering
- Multi-day forecast generation and visualization
- Real-time market conditions display
- Interactive Plotly graphs for price forecasts
- Dark mode theme with modern UI

**Key Features**:
- Uses `enhanced_predictor.py` for predictions
- Displays forecasts with historical context
- Shows data source indicators (e-NAM primary, fallback)
- Progress indicators for data fetching
- Error handling with user-friendly messages

**Usage**:
```bash
streamlit run app.py
```

**Dependencies**: `enhanced_predictor`, `geolocation_fetcher`, `get_available_commodities`, `location_normalizer`

---

## 🧠 Core Prediction Engine

### `enhanced_predictor.py`
**Purpose**: Core prediction engine with weather integration, market conditions, and calibration

**Functionality**:
- Single price predictions with calibration
- Multi-day forecast generation (7-90 days)
- Current market conditions fetching (5 sources)
- Prediction calibration based on current market prices
- Previous week data fallback
- Weather data integration
- Advanced feature engineering

**Key Functions**:
- `predict_price()`: Single prediction with calibration
- `predict_with_forecast()`: Multi-day forecasts
- `fetch_current_market_conditions()`: Market data from 5 sources
- `calibrate_prediction()`: Market-aligned prediction adjustment
- `_fetch_previous_week_data()`: Fallback mechanism

**Usage Context**: Called by `app.py` for all predictions. This is the main entry point for the prediction system.

**Dependencies**: `enam_fetcher`, `apmc_mapper`, `weather_data_fetcher`, `geolocation_fetcher`, `location_normalizer`, `phase2_feature_engineering`

---

### `production_predictor.py`
**Purpose**: Production-ready predictor with full feature engineering (alternative to enhanced_predictor)

**Functionality**:
- Uses Phase 2 feature engineering
- Loads Phase 3 trained models
- Batch prediction support
- Caching for performance
- Model selection (lightgbm, xgboost, ensemble)

**Usage Context**: Alternative predictor for production use. Can be used for batch processing or API integration.

**Dependencies**: `phase2_feature_engineering`, `phase3_model_development`, `get_available_commodities`

---

## 📡 Data Fetching Modules

### `enam_fetcher.py`
**Purpose**: Fetches market data from e-NAM (National Agriculture Market) - PRIMARY data source

**Functionality**:
- Fetches from 3 e-NAM endpoints:
  1. Trade Data: `/web/dashboard/trade-data`
  2. Agm_Enam_ctrl: `/web/dashboard/Agm_Enam_ctrl`
  3. Live Price: `/web/dashboard/live_price`
- Automatic APMC mapping to districts/states
- Multiple fetch strategies (AJAX, HTML scraping)
- Data standardization

**Key Class**: `EnamFetcher`
- `fetch_live_prices()`: Main fetch function
- `_fetch_trade_data()`: Trade data endpoint
- `_fetch_agm_enam_data()`: Agm_Enam_ctrl endpoint
- `_scrape_live_prices()`: HTML scraping fallback
- `_standardize_dataframe()`: Data cleaning and APMC mapping

**Usage Context**: Called by `enhanced_predictor._fetch_enam_wrapper()` as PRIMARY source.

**Dependencies**: `apmc_mapper` (for APMC mapping)

---

### `apmc_mapper.py`
**Purpose**: Maps APMC (Agricultural Produce Market Committee) names to districts and states

**Functionality**:
- Automatic APMC to district/state mapping
- Multiple mapping strategies:
  1. Name extraction (e.g., "Chennai APMC" → Chennai, Tamil Nadu)
  2. Geocoding (OpenStreetMap API)
  3. Fuzzy matching with known districts
  4. State-constrained search
- Persistent caching (`data/apmc_mapping.json`)
- Validates and enhances existing mappings

**Key Class**: `APMCMapper`
- `map_apmc()`: Map single APMC
- `map_apmc_dataframe()`: Map APMCs in DataFrame
- `build_mapping_from_enam()`: Build mapping from e-NAM data

**Usage Context**: Automatically called by `enam_fetcher._standardize_dataframe()` when APMC column is detected.

**Dependencies**: `location_normalizer`, `enam_fetcher`, `cache_manager`

---

### `commodityonline_fetcher.py`
**Purpose**: Fetches market prices from Commodity Online website

**Functionality**:
- Scrapes mandi prices from `https://www.commodityonline.com/mandiprices`
- Multiple strategies (API, AJAX, HTML scraping)
- Retry logic with exponential backoff
- Data standardization

**Key Class**: `CommodityOnlineFetcher`
- `fetch_mandi_prices()`: Main fetch function

**Usage Context**: Called by `enhanced_predictor._fetch_commodityonline_wrapper()` as secondary source.

---

### `ncdex_fetcher.py`
**Purpose**: Fetches spot prices from NCDEX (National Commodity & Derivatives Exchange)

**Functionality**:
- Fetches spot prices from `https://www.ncdex.com/markets/spotprices`
- Commodity-level price data
- Historical spot price data

**Key Class**: `NCDEXFetcher`
- `fetch_spot_prices()`: Main fetch function

**Usage Context**: Called by `enhanced_predictor._fetch_ncdex_wrapper()` as secondary source.

---

### `market_data_fetcher.py`
**Purpose**: Fetches market data from AGMARKNET (Government of India portal)

**Functionality**:
- Fetches from `https://agmarknet.gov.in`
- Multiple strategies (API, Selenium, HTML scraping)
- Handles dynamic content loading
- Data standardization

**Key Functions**:
- `fetch_agmarknet_data()`: Main fetch function

**Usage Context**: Called by `enhanced_predictor._fetch_agmarknet_wrapper()` as secondary source.

---

### `enhanced_market_data_fetcher.py`
**Purpose**: Enhanced fetcher for Data.gov.in API and other government data sources

**Functionality**:
- Fetches from Data.gov.in API
- Handles API authentication
- Data standardization
- Multiple data source support

**Key Functions**:
- `fetch_data_gov_data()`: Data.gov.in API fetch
- `fetch_enam_data()`: Alternative e-NAM fetch
- `fetch_msamb_data()`: MSAMB data fetch

**Usage Context**: Called by `enhanced_predictor._fetch_gov_data_wrapper()` for government data sources.

---

### `agmarknet_api_fetcher.py`
**Purpose**: API-based fetcher for AGMARKNET 2.0

**Functionality**:
- Uses AGMARKNET 2.0 API
- More reliable than web scraping
- Structured data format

**Usage Context**: Called by `market_data_fetcher.py` as primary method.

---

## 🌤️ Weather and Location

### `weather_data_fetcher.py`
**Purpose**: Fetches weather data for agricultural predictions

**Functionality**:
- Fetches from Open-Meteo API
- Historical and forecast weather data
- Temperature, precipitation, humidity
- Location-based weather fetching

**Key Functions**:
- `fetch_weather_data()`: Main fetch function

**Usage Context**: Called by `enhanced_predictor.py` for weather features in predictions.

---

### `geolocation_fetcher.py`
**Purpose**: Fetches latitude/longitude coordinates for locations

**Functionality**:
- Uses Nominatim (OpenStreetMap) API
- Converts state/district names to coordinates
- Caching for performance
- Location normalization before geocoding

**Key Functions**:
- `get_coordinates()`: Get lat/lon for state/district

**Usage Context**: Called by `enhanced_predictor.py` and `weather_data_fetcher.py` to get coordinates for weather data.

**Dependencies**: `location_normalizer`, `cache_manager`

---

## 🔧 Utility Modules

### `location_normalizer.py`
**Purpose**: Normalizes state and district names to match training data

**Functionality**:
- Handles name variations (e.g., "TN" → "Tamil Nadu")
- Handles old/new names (e.g., "Gurgaon" → "Gurugram")
- State-aware district normalization
- Fuzzy matching for similar names

**Key Functions**:
- `normalize_state_name()`: Normalize state names
- `normalize_district_name()`: Normalize district names
- `normalize_location()`: Normalize both
- `get_all_states()`: Get available states
- `get_all_districts()`: Get available districts

**Usage Context**: Used throughout the system to ensure location names match training data format.

---

### `get_available_commodities.py`
**Purpose**: Provides location-commodity mapping based on training data

**Functionality**:
- Returns available states, districts, commodities
- Validates state-district-commodity combinations
- Filters based on actual training data
- Location information retrieval

**Key Functions**:
- `get_available_states()`: Get all states in training data
- `get_available_districts(state)`: Get districts for a state
- `get_available_commodities(state, district)`: Get commodities for location
- `is_valid_combination()`: Validate combination
- `get_location_info()`: Get location metadata

**Usage Context**: Used by `app.py` to filter dropdown options and validate user input.

**Dependencies**: `location_normalizer`

---

### `cache_manager.py`
**Purpose**: In-memory caching system for performance optimization

**Functionality**:
- TTL-based caching (Time To Live)
- Different TTLs for different data types
- Cache invalidation
- Memory-efficient storage

**Key Class**: `CacheManager`
- `get()`: Retrieve cached data
- `set()`: Store data in cache
- `clear()`: Clear cache

**Usage Context**: Used by data fetchers and predictors to cache frequently accessed data.

---

### `error_handler.py`
**Purpose**: Centralized error handling utilities

**Functionality**:
- Standardized error handling
- Error logging
- User-friendly error messages

**Usage Context**: Used throughout the system for consistent error handling.

---

## 🔄 ML Pipeline Files

### `phase1_data_analysis.py`
**Purpose**: Phase 1 - Data analysis and exploration

**Functionality**:
- Loads and analyzes datasets
- Generates statistics and visualizations
- Identifies patterns (seasonal, geographic, commodity)
- Data quality checks
- Missing data analysis

**Key Class**: `DataAnalyzer`
- Analyzes data distributions
- Creates visualizations
- Generates analysis reports

**Usage Context**: Run during initial data analysis phase. Generates insights for feature engineering.

**Run**: `python phase1_data_analysis.py`

---

### `phase2_feature_engineering.py`
**Purpose**: Phase 2 - Advanced feature engineering (86 features)

**Functionality**:
- Creates 86 engineered features:
  - Temporal features (date, season, crop seasons)
  - Price features (lags, moving averages, volatility)
  - Location features (state/district averages)
  - Commodity features (crop statistics)
  - Weather features (temperature, rainfall)
  - Interaction features
- Feature normalization
- Missing value handling

**Key Class**: `FeatureEngineer`
- `create_temporal_features()`: Date/season features
- `create_price_features()`: Price lags and statistics
- `create_location_features()`: Location-based features
- `create_commodity_features()`: Commodity-based features
- `create_weather_features()`: Weather features
- `engineer_all_features()`: Create all features

**Usage Context**: Used by model training and prediction to create features from raw data.

**Dependencies**: Pandas, NumPy

---

### `phase3_model_development.py`
**Purpose**: Phase 3 - Model training and development

**Functionality**:
- Trains multiple model types:
  - Baseline models (historical average, last value, moving average)
  - Tree-based (XGBoost, LightGBM)
  - Neural networks (Feedforward, LSTM, GRU)
  - Ensemble models
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Model saving

**Key Class**: `ModelTrainer`
- `train_baseline_models()`: Train baseline models
- `train_tree_models()`: Train XGBoost/LightGBM
- `train_neural_networks()`: Train neural networks
- `train_ensemble()`: Train ensemble models
- `save_models()`: Save trained models

**Usage Context**: Run to train models. Models saved to `models/phase3/`.

**Run**: `python phase3_model_development.py`

**Dependencies**: `phase2_feature_engineering`, XGBoost, LightGBM, TensorFlow (optional)

---

### `phase4_model_evaluation.py`
**Purpose**: Phase 4 - Model evaluation and metrics

**Functionality**:
- Comprehensive model evaluation
- Metrics calculation (MAE, RMSE, MAPE, R²)
- Error analysis
- Feature importance analysis
- SHAP interpretability
- Model comparison

**Key Class**: `ModelEvaluator`
- Evaluates models on test set
- Generates evaluation reports
- Creates visualizations

**Usage Context**: Run after model training to evaluate performance.

**Run**: `python phase4_model_evaluation.py`

---

### `phase5_system_integration.py`
**Purpose**: Phase 5 - System integration and deployment

**Functionality**:
- Integrates all components
- API integration
- Model versioning
- System testing
- Deployment preparation

**Usage Context**: Final integration phase before deployment.

**Run**: `python phase5_system_integration.py`

---

### `run_all_phases.py`
**Purpose**: Runs all 5 phases sequentially

**Functionality**:
- Executes phases 1-5 in order
- Complete ML pipeline from data analysis to deployment
- Progress tracking
- Error handling

**Usage Context**: Run to execute complete ML pipeline from scratch.

**Run**: `python run_all_phases.py`

---

## 🔄 Retraining and Maintenance

### `model_retraining_pipeline.py`
**Purpose**: Automated model retraining pipeline

**Functionality**:
- Checks if retraining is needed (based on last training date)
- Backs up existing models
- Loads latest data
- Retrains models
- Evaluates new models
- Updates model versions

**Key Class**: `ModelRetrainingPipeline`
- `check_retrain_needed()`: Check if retraining required
- `backup_existing_models()`: Backup before retraining
- `retrain_models()`: Retrain all models

**Usage Context**: Run periodically (e.g., monthly) to keep models updated with new data.

**Run**: `python model_retraining_pipeline.py`

---

### `retrain_with_consolidated_data.py`
**Purpose**: Retrain models with consolidated data

**Functionality**:
- Loads consolidated dataset
- Feature engineering
- Model training
- Model evaluation
- Saves updated models

**Usage Context**: Run when consolidated data is updated.

**Run**: `python retrain_with_consolidated_data.py`

---

### `retrain_with_weather_data.py`
**Purpose**: Retrain models with weather-enriched data

**Functionality**:
- Enriches data with weather features
- Trains weather-aware models
- Saves to `models/with_weather/`

**Usage Context**: Run to create weather-enriched models.

**Run**: `python retrain_with_weather_data.py`

---

## 📊 Data Processing

### `preprocessing.py`
**Purpose**: Data preprocessing utilities

**Functionality**:
- Data cleaning
- Missing value handling
- Data type conversion
- Market and weather data merging
- Feature creation (basic)

**Key Functions**:
- `preprocess_market_data()`: Clean market data
- `merge_datasets()`: Merge market and weather data
- `create_features()`: Create basic features

**Usage Context**: Used in data preparation pipeline.

---

### `consolidate_location_data.py`
**Purpose**: Consolidates location data from multiple sources

**Functionality**:
- Combines location data
- Removes duplicates
- Standardizes location names
- Creates location mappings

**Usage Context**: Run to consolidate location data before training.

**Run**: `python consolidate_location_data.py`

---

### `enrich_data_with_weather.py`
**Purpose**: Enriches market data with weather information

**Functionality**:
- Fetches weather data for locations
- Merges weather with market data
- Creates weather features
- Saves enriched dataset

**Usage Context**: Run to create weather-enriched training data.

**Run**: `python enrich_data_with_weather.py`

---

### `run_weather_enrichment.py`
**Purpose**: Runs weather enrichment process

**Functionality**:
- Orchestrates weather data fetching
- Enriches market data
- Saves enriched datasets

**Usage Context**: Wrapper script for weather enrichment.

**Run**: `python run_weather_enrichment.py`

---

### `process_custom_datasets.py`
**Purpose**: Processes custom datasets for training

**Functionality**:
- Loads custom datasets
- Standardizes format
- Merges with existing data
- Prepares for training

**Usage Context**: Run when adding new data sources.

**Run**: `python process_custom_datasets.py`

---

### `kaggle_data_fetcher.py`
**Purpose**: Fetches and processes Kaggle datasets

**Functionality**:
- Downloads Kaggle datasets
- Processes and standardizes
- Merges multiple Kaggle sources
- Creates combined dataset

**Usage Context**: Run to fetch Kaggle datasets for training.

**Run**: `python kaggle_data_fetcher.py`

---

## 📈 Batch Processing and Monitoring

### `batch_prediction.py`
**Purpose**: Batch prediction system for multiple predictions

**Functionality**:
- Processes multiple predictions efficiently
- Commodity-level batch predictions
- Location-level batch predictions
- Progress tracking
- Results export

**Key Class**: `BatchPredictor`
- `predict_for_commodity()`: Predict for commodity across locations
- `predict_for_location()`: Predict for location across commodities
- `predict_batch()`: General batch prediction

**Usage Context**: Use for generating predictions for multiple commodities/locations at once.

**Usage**:
```python
from batch_prediction import BatchPredictor

batch = BatchPredictor()
predictions = batch.predict_for_commodity('Wheat', start_date='2025-01-01', end_date='2025-01-31')
```

---

### `monitoring_system.py`
**Purpose**: Prediction monitoring and statistics

**Functionality**:
- Tracks prediction requests
- Monitors prediction accuracy
- Generates statistics
- Error tracking
- Performance metrics

**Key Class**: `PredictionMonitor`
- `log_prediction()`: Log prediction request
- `get_statistics()`: Get monitoring statistics
- `get_error_rate()`: Get error rate

**Usage Context**: Used for production monitoring and analytics.

**Usage**:
```python
from monitoring_system import get_monitor

monitor = get_monitor()
stats = monitor.get_statistics(days=30)
```

---

## 🔍 Analysis and Utilities

### `show_model_metrics.py`
**Purpose**: Displays model performance metrics

**Functionality**:
- Loads model metadata
- Displays performance metrics
- Compares models
- Shows feature importance
- Generates reports

**Usage Context**: Run to view model performance after training.

**Run**: `python show_model_metrics.py`

---

### `check_training_coverage_detailed.py`
**Purpose**: Checks training data coverage

**Functionality**:
- Analyzes training data coverage
- Identifies gaps (missing commodities/locations)
- Generates coverage reports
- Suggests data collection priorities

**Usage Context**: Run to understand training data completeness.

**Run**: `python check_training_coverage_detailed.py`

---

### `simulate_user_interaction.py`
**Purpose**: Simulates user interactions for testing

**Functionality**:
- Tests prediction system with sample inputs
- Verifies predictions work correctly
- Tests error handling
- Generates test reports

**Usage Context**: Run for system testing and validation.

**Run**: `python simulate_user_interaction.py`

---

## 📁 Configuration Files

### `requirements.txt`
**Purpose**: Python package dependencies

**Functionality**:
- Lists all required Python packages
- Version specifications
- Installation instructions

**Usage**:
```bash
pip install -r requirements.txt
```

---

### `README.md`
**Purpose**: Main project documentation

**Functionality**:
- Project overview
- Quick start guide
- Usage examples
- Feature documentation
- Troubleshooting

**Usage Context**: Primary documentation for users and developers.

---

## 📂 Directory Structure

### `data/`
**Purpose**: Data storage directory

**Subdirectories**:
- `combined/`: Combined datasets (1.26M records)
- `processed/`: Feature-engineered data
- `kaggle_combined/`: Kaggle datasets
- `mappings/`: Location/crop mappings
- `apmc_mapping.json`: APMC mappings cache

---

### `models/`
**Purpose**: Trained model storage

**Subdirectories**:
- `phase3/`: Main trained models (XGBoost, LightGBM, Neural Networks)
- `consolidated/`: Consolidated data models
- `with_weather/`: Weather-enriched models
- `models_advanced/`: Commodity-level models
- `models_by_commodity/`: Commodity-specific models
- `models_by_location_commodity/`: Location-commodity models

---

### `tests/`
**Purpose**: Unit and integration tests

**Files**:
- `test_app.py`: Streamlit app tests
- `test_integration.py`: Integration tests
- `test_predictor.py`: Predictor tests
- `test_train_model.py`: Model training tests
- `test_preprocessing.py`: Preprocessing tests
- `test_geolocation_fetcher.py`: Geocoding tests
- `test_weather_data_fetcher.py`: Weather fetcher tests
- `test_market_data_fetcher.py`: Market data fetcher tests

**Usage**:
```bash
python -m pytest tests/
```

---

## 🔗 File Dependencies

### Core Dependencies Flow

```
app.py
    ↓
enhanced_predictor.py
    ├─→ enam_fetcher.py → apmc_mapper.py
    ├─→ weather_data_fetcher.py → geolocation_fetcher.py
    ├─→ commodityonline_fetcher.py
    ├─→ ncdex_fetcher.py
    ├─→ enhanced_market_data_fetcher.py
    ├─→ market_data_fetcher.py
    ├─→ location_normalizer.py
    ├─→ get_available_commodities.py
    ├─→ cache_manager.py
    └─→ phase2_feature_engineering.py
```

### ML Pipeline Flow

```
phase1_data_analysis.py
    ↓
phase2_feature_engineering.py
    ↓
phase3_model_development.py
    ↓
phase4_model_evaluation.py
    ↓
phase5_system_integration.py
```

---

## 📝 Usage Patterns

### For End Users
- **Primary File**: `app.py` (Streamlit application)
- **Run**: `streamlit run app.py`

### For Developers
- **Core Prediction**: `enhanced_predictor.py`
- **Data Fetching**: `enam_fetcher.py`, `commodityonline_fetcher.py`, etc.
- **Feature Engineering**: `phase2_feature_engineering.py`
- **Model Training**: `phase3_model_development.py`

### For Data Scientists
- **Data Analysis**: `phase1_data_analysis.py`
- **Feature Engineering**: `phase2_feature_engineering.py`
- **Model Development**: `phase3_model_development.py`
- **Evaluation**: `phase4_model_evaluation.py`

### For System Administrators
- **Retraining**: `model_retraining_pipeline.py`
- **Monitoring**: `monitoring_system.py`
- **Batch Processing**: `batch_prediction.py`

---

## 🎯 Key Integration Points

### 1. Prediction Flow
```
app.py → enhanced_predictor.py → [data fetchers] → [models] → predictions
```

### 2. Data Fetching Flow
```
enhanced_predictor.py → _fetch_enam_wrapper() → enam_fetcher.py → apmc_mapper.py
```

### 3. Feature Engineering Flow
```
enhanced_predictor.py → phase2_feature_engineering.py → 86 features
```

### 4. Model Loading Flow
```
enhanced_predictor.py → load_model() → models/phase3/*.pkl
```

---

## 📚 Additional Notes

### File Naming Conventions
- `*_fetcher.py`: Data fetching modules
- `phase*.py`: ML pipeline phases
- `*_predictor.py`: Prediction modules
- `*_mapper.py`: Mapping/transformation modules
- `*_normalizer.py`: Normalization modules

### Module Organization
- **Production**: Core application files
- **Data**: Data fetching and processing
- **ML**: Machine learning pipeline
- **Utils**: Utility and helper modules
- **Tests**: Test files

---

**Last Updated**: December 2024  
**Maintained By**: Project Team

