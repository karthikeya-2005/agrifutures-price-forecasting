"""
Master script to run weather enrichment and model retraining
This script orchestrates the complete process
"""
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run weather enrichment and retraining"""
    logger.info("="*80)
    logger.info("WEATHER ENRICHMENT AND MODEL RETRAINING")
    logger.info("="*80)
    
    # Step 1: Enrich data with weather
    logger.info("\n" + "="*80)
    logger.info("STEP 1: ENRICHING DATA WITH WEATHER")
    logger.info("="*80)
    
    enriched_file = Path('data/combined/all_sources_consolidated_with_weather.csv')
    
    if not enriched_file.exists():
        logger.info("Weather-enriched data not found. Running enrichment...")
        logger.info("NOTE: This may take several hours depending on data size and API limits.")
        
        response = input("Do you want to proceed with weather enrichment? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Skipping weather enrichment. Using existing consolidated data.")
        else:
            try:
                result = subprocess.run(
                    [sys.executable, 'enrich_data_with_weather.py'],
                    check=True,
                    capture_output=False
                )
                logger.info("Weather enrichment completed successfully!")
            except subprocess.CalledProcessError as e:
                logger.error(f"Weather enrichment failed: {e}")
                logger.info("Continuing with existing data...")
    else:
        logger.info(f"Weather-enriched data already exists: {enriched_file}")
        logger.info("Skipping enrichment step.")
    
    # Step 2: Retrain models
    logger.info("\n" + "="*80)
    logger.info("STEP 2: RETRAINING MODELS WITH WEATHER DATA")
    logger.info("="*80)
    
    response = input("Do you want to retrain models? This will take some time. (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Skipping model retraining.")
        return
    
    try:
        result = subprocess.run(
            [sys.executable, 'retrain_with_weather_data.py'],
            check=True,
            capture_output=False
        )
        logger.info("Model retraining completed successfully!")
        logger.info("New models saved to: models/with_weather/")
    except subprocess.CalledProcessError as e:
        logger.error(f"Model retraining failed: {e}")
        return
    
    logger.info("\n" + "="*80)
    logger.info("PROCESS COMPLETE")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("1. Test predictions with the new models")
    logger.info("2. Verify weather features are being used")
    logger.info("3. Check model performance improvements")

if __name__ == "__main__":
    main()

