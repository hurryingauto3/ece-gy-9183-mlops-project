from locust import HttpUser, task, between
from datetime import datetime, timedelta
import random

class PredictUser(HttpUser):
    # wait_time = between(1, 5)  # Users wait 1-5 seconds between tasks
    host = "http://model-serving:8000" # Target model-serving by its service name

    @task
    def get_single_prediction(self):
        # Define a list of sample FIPS codes, crops, and a base date
        sample_fips = ["19153", "17031", "DUMMY"] # Add more valid FIPS if you have them
        sample_crops = ["corn", "soybeans"]
        base_year = datetime.now().year - 1 # Use last year for more likely data

        # Construct a somewhat realistic payload
        selected_fips = random.choice(sample_fips)
        selected_crop = random.choice(sample_crops)
        # Generate a random cut-off date within a reasonable range for the year
        start_date = datetime(base_year, 4, 1) # April 1st
        end_date = datetime(base_year, 10, 30) # October 30th
        random_days = random.randint(0, (end_date - start_date).days)
        cut_off_date_str = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        # Define histogram bins (ensure this matches what your model expects)
        histogram_bins = [0, 50, 100, 120, 140, 160, 180, 200, 220, 240, 260, 300]
        if selected_crop == "soybeans":
            histogram_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]

        payload = {
            "county": selected_fips,
            "year": base_year,
            "cut_off_date": cut_off_date_str,
            "crop": selected_crop,
            "histogram_bins": histogram_bins
        }

        self.client.post("/predict", json=payload, name="/predict (model-serving)")


class FeatureServiceUser(HttpUser):
    # wait_time = between(1, 3) # Users wait 1-3 seconds between tasks
    host = "http://feature-service:8001" # Target feature-service by its service name

    # Sample data for feature service requests
    sample_fips = ["19153", "17031", "DUMMY", "01001", "06037"] # Added more diverse FIPS
    sample_crops = ["corn", "soybeans", "wheat"]
    base_year = datetime.now().year - 1

    @task
    def get_features(self):
        selected_fips = random.choice(self.sample_fips)
        selected_crop = random.choice(self.sample_crops)
        
        # Generate a random cut-off date, or sometimes no cut-off date
        cut_off_date_param = None
        if random.random() < 0.7: # 70% chance of having a cut-off date
            start_date = datetime(self.base_year, 4, 1)
            end_date = datetime(self.base_year, 10, 30)
            random_days = random.randint(0, (end_date - start_date).days)
            cut_off_date_param = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

        params = {
            "county": selected_fips,
            "year": self.base_year,
            "crop": selected_crop # Crop is optional for feature service but good to include if model uses it
        }
        if cut_off_date_param:
            params["cut_off_date"] = cut_off_date_param

        self.client.get("/features", params=params, name="/features (feature-serving)")

# To run this with Docker Compose (after adding the service):
# docker-compose --profile testing up locust-service
# Then open your browser to http://localhost:8089 (or the mapped port) 