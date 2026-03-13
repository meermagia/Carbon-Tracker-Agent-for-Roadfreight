import random
import requests
from datetime import datetime, timedelta

API_URL = "http://localhost:8000/api/v1/ingest_shipment"
NUM_SHIPMENTS = 400

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Hyderabad", "Pune", "Kolkata", "Ahmedabad"
]

VEHICLE_TYPES = ["diesel_truck", "electric_truck"]


def random_date_within_last_90_days():
    today = datetime.now()
    random_days = random.randint(0, 90)
    return (today - timedelta(days=random_days)).strftime("%Y-%m-%d")


def generate_shipment():
    origin = random.choice(CITIES)
    destination = random.choice(CITIES)

    while destination == origin:
        destination = random.choice(CITIES)

    return {
    "shipment_id": f"SHP-{random.randint(100000, 999999)}",
    "origin_location": origin,
    "destination_location": destination,
    "distance_km": random.randint(100, 2000),
    "weight_tons": round(random.uniform(1, 20), 2),
    "transport_mode": random.choice(VEHICLE_TYPES),  # <-- CHANGED
    "shipment_date": random_date_within_last_90_days()
}

def main():
    success_count = 0

    for i in range(NUM_SHIPMENTS):
        shipment = generate_shipment()

        try:
            response = requests.post(API_URL, json=shipment)

            if response.status_code in [200, 201]:
                success_count += 1
                if (i + 1) % 50 == 0:
                    print(f"✅ Inserted {i + 1} shipments...")
            else:
                print(f"❌ Failed at shipment {i + 1}")
                print("Status:", response.status_code)
                print("Response:", response.text)

        except Exception as e:
            print("❌ Exception occurred:", str(e))

    print("\n---------------------------------")
    print(f"Finished. Successfully inserted {success_count} shipments.")
    print("---------------------------------")


if __name__ == "__main__":
    main()
    