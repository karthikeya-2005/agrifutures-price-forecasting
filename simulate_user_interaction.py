"""
Simulate user interactions with the application
Tests various inputs and verifies outputs
"""
from enhanced_predictor import predict_with_forecast
from datetime import datetime, timedelta
import sys

def test_interaction(state, district, crop, days_ahead):
    """Simulate a user interaction with the app"""
    print(f"\n{'='*80}")
    print(f"USER INTERACTION: {crop} in {district}, {state}")
    print(f"{'='*80}")
    print(f"Input:")
    print(f"  State: {state}")
    print(f"  District: {district}")
    print(f"  Crop: {crop}")
    print(f"  Forecast Period: {days_ahead} days")
    
    target_date = datetime.now().date() + timedelta(days=days_ahead)
    
    def progress_callback(status, step):
        print(f"  -> {status}")
    
    print(f"\nGenerating prediction...")
    result = predict_with_forecast(
        state=state,
        district=district,
        crop=crop,
        target_date=target_date,
        days_ahead=days_ahead,
        progress_callback=progress_callback
    )
    
    if result:
        predictions = result.get('predictions', [])
        market_conditions = result.get('market_conditions', {})
        price_unit = result.get('price_unit_display', 'N/A')
        quantity_unit = result.get('quantity_unit', 'N/A')
        
        print(f"\n[SUCCESS] Prediction Generated!")
        print(f"\nOutput:")
        print(f"  Predictions: {len(predictions)}")
        print(f"  Price Unit: {price_unit}")
        print(f"  Quantity: {quantity_unit} (1 {quantity_unit} = 100 kg)")
        
        if predictions:
            print(f"\n  Forecast Prices:")
            for pred in predictions:
                date = pred.get('date', 'N/A')
                price = pred.get('price', 0)
                weather = pred.get('weather', {})
                print(f"    {date}: Rs {price:,.2f} {price_unit}")
                if weather:
                    temp = weather.get('temperature_2m_max', 'N/A')
                    rain = weather.get('precipitation_sum', 'N/A')
                    print(f"      Weather: Temp {temp}C, Rain {rain}mm")
        
        if market_conditions and market_conditions.get('current_price'):
            print(f"\n  Current Market Price: Rs {market_conditions['current_price']:,.2f} {price_unit}")
        else:
            print(f"\n  Current Market Price: Not available (using historical data)")
        
        # Verify output quality
        checks = {
            "Has predictions": len(predictions) > 0,
            "Has price unit": price_unit != 'N/A',
            "Has quantity unit": quantity_unit != 'N/A',
            "Prices are positive": all(p.get('price', 0) > 0 for p in predictions),
            "Dates are valid": all(p.get('date') for p in predictions)
        }
        
        print(f"\n  Verification:")
        all_passed = True
        for check, passed in checks.items():
            status = "[OK]" if passed else "[FAIL]"
            print(f"    {status} {check}")
            if not passed:
                all_passed = False
        
        return all_passed
    else:
        print(f"\n[FAILED] No result returned")
        return False

# Test multiple interactions
print("="*80)
print("SIMULATING USER INTERACTIONS WITH APPLICATION")
print("="*80)

test_cases = [
    ("Kerala", "Thiruvananthapuram", "Rice", 3),
    ("Punjab", "Amritsar", "Wheat", 5),
    ("Maharashtra", "Pune", "Tomato", 7)
]

results = []
for state, district, crop, days in test_cases:
    try:
        passed = test_interaction(state, district, crop, days)
        results.append((f"{crop} in {district}, {state}", passed))
    except Exception as e:
        print(f"\n[ERROR] {e}")
        results.append((f"{crop} in {district}, {state}", False))

# Summary
print(f"\n{'='*80}")
print("INTERACTION SUMMARY")
print(f"{'='*80}")

for name, passed in results:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {name}")

passed_count = sum(1 for _, p in results if p)
print(f"\n{passed_count}/{len(results)} interactions successful")

if passed_count == len(results):
    print("\n[SUCCESS] All interactions working correctly!")
    sys.exit(0)
else:
    print("\n[WARNING] Some interactions had issues")
    sys.exit(1)

