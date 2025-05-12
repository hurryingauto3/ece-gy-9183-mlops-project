import streamlit as st
import pandas as pd
# import numpy as np # No longer explicitly needed for placeholders in this version
import plotly.express as px # Keep for potential other plots
from datetime import datetime, date # date is used for date_input

# Import new and existing services/processors
from api_service import fetch_single_crop_histogram_prediction #, fetch_historical_data # Keep historical if needed later
from data_processor import process_histogram_prediction #, process_historical_data, combine_historical_and_prediction

# Prometheus client
from prometheus_client import Counter, Gauge, start_http_server
import threading # To run Prometheus server in a separate thread
import atexit # To handle cleanup if needed, though less critical for read-only metrics server

# --- Prometheus Metrics Setup ---
# Ensure this block runs only once
PROMETHEUS_PORT = 9095 # Internal port for Prometheus scraping within Docker network
_prometheus_server_started = False
_prometheus_thread = None

# Define metrics
PAGE_VIEWS = Counter('dashboard_page_views', 'Dashboard page view count')
PREDICTION_REQUESTS = Counter('dashboard_prediction_requests_total', 'Total prediction requests made from dashboard')
ACTIVE_SESSIONS = Gauge('dashboard_active_sessions', 'Number of active dashboard user sessions') # Simplified
LAST_PREDICTION_TIME = Gauge('dashboard_last_prediction_timestamp_seconds', 'Timestamp of the last prediction request')

def start_metrics_server():
    global _prometheus_server_started, _prometheus_thread
    if not _prometheus_server_started:
        try:
            start_http_server(PROMETHEUS_PORT)
            _prometheus_server_started = True
            st.session_state['_prometheus_server_started'] = True # Use session state to persist across reruns
            print(f"INFO: Prometheus metrics server started on port {PROMETHEUS_PORT}")
        except Exception as e:
            print(f"ERROR: Failed to start Prometheus metrics server on port {PROMETHEUS_PORT}: {e}")

# Attempt to start the server if not already started in this session
if '_prometheus_server_started' not in st.session_state:
    # For multipage apps or more complex scenarios, managing the thread might be needed.
    # For a single page app, starting it once should be okay.
    # A more robust way for multipage apps is to manage a single thread across sessions.
    _prometheus_thread = threading.Thread(target=start_metrics_server, daemon=True)
    _prometheus_thread.start()

# Increment page view counter (this will run on every script run / interaction)
PAGE_VIEWS.inc()
# For active sessions, Streamlit doesn't have a direct hook for session end without more complex management.
# This gauge would ideally be managed by a more sophisticated session tracking mechanism if needed.
# For simplicity, we might increment on new session and decrement if we could detect session close.
# A simpler proxy: set to number of current unique session IDs (if accessible and meaningful).

# Page configuration
st.set_page_config(
    page_title="Crop Yield Histogram Prediction Dashboard",
    page_icon="🌾",
    layout="wide"
)

# Set title and description
st.title("🌾 Crop Yield Histogram Prediction")
st.markdown("""
    Select a FIPS county code, cut-off date, and crop type to visualize the predicted yield histogram.
""")

# Sidebar for input controls
st.sidebar.header("Prediction Inputs")

# FIPS County Code input
# In a real app, this could be validated, fetched from a list, or a map selector.
county_fips_input = st.sidebar.text_input(
    "FIPS County Code",
    value="DUMMY"  # Default to DUMMY FIPS for easy testing
)

# Cut-off Date input
cut_off_date_input = st.sidebar.date_input(
    "Cut-off Date",
    value=date(datetime.now().year, 7, 15) # Default to July 15 of current year
)

# Crop Type selection
# Common crops, could be expanded or made dynamic
crop_options = ["corn", "soybeans", "wheat", "rice", "cotton", "barley"]
selected_crop = st.sidebar.selectbox(
    "Select Crop Type",
    crop_options,
    index=0  # Default to the first crop (corn)
)

# Histogram Bins (fixed for now, could be made a user input later)
# This should match the bins your model is trained for or expects.
# For the dummy model, which outputs 5 probabilities, we need 6 bin edges.
# fixed_histogram_bins = [0, 50, 100, 150, 200, 250] # 6 edges = 5 bins
# Example for corn: [0, 50, 100, 120, 140, 160, 180, 200, 220, 240, 260, 300, 350]

# st.sidebar.markdown("---_Note: Histogram bins are currently fixed for 5-bin output._---")

histogram_bins_str = st.sidebar.text_input(
    "Histogram Bin Edges (comma-separated)",
    value="0, 50, 100, 150, 200, 250", # Default for 5 bins
    help="Enter comma-separated numbers for bin edges, e.g., 0,20,40,60,80,100. Must be at least 2 edges, strictly increasing."
)
st.sidebar.markdown("---_Note: The number of bins implied (edges - 1) must match the model's output._---")


# --- Main app layout ---
if st.sidebar.button("Get Prediction"):
    user_histogram_bins = []
    valid_bins = False
    if not county_fips_input:
        st.error("Please enter a FIPS County Code.")
    elif not cut_off_date_input:
        st.error("Please select a Cut-off Date.")
    elif not selected_crop:
        st.error("Please select a Crop Type.")
    else:
        try:
            user_histogram_bins = [float(b.strip()) for b in histogram_bins_str.split(',') if b.strip()]
            if len(user_histogram_bins) < 2:
                st.error("Histogram bins must contain at least two edges.")
            elif not all(user_histogram_bins[i] < user_histogram_bins[i+1] for i in range(len(user_histogram_bins)-1)):
                st.error("Histogram bin edges must be strictly increasing values.")
            else:
                valid_bins = True
        except ValueError:
            st.error("Invalid input for histogram bin edges. Please enter comma-separated numbers (e.g., 0,50,100).")
        except Exception as e:
            st.error(f"Error processing histogram bins: {e}")

    if valid_bins: # Proceed only if all inputs including bins are valid
        PREDICTION_REQUESTS.inc() # Increment counter for prediction requests
        LAST_PREDICTION_TIME.set_to_current_time()
        with st.spinner(f"Fetching prediction for {selected_crop} in {county_fips_input} for date {cut_off_date_input}..."):
            try:
                # API call to get histogram prediction
                api_response = fetch_single_crop_histogram_prediction(
                    county_fips=county_fips_input,
                    cut_off_date=cut_off_date_input,
                    crop_name=selected_crop,
                    histogram_bins=user_histogram_bins # Use parsed user input
                )
                
                if api_response and api_response.get("predicted_histogram"):
                    st.subheader(f"Predicted Yield Histogram for {selected_crop.capitalize()}")
                    st.markdown(f"**County FIPS:** `{api_response.get('county', county_fips_input)}` | "
                                f"**Cut-off Date:** `{api_response.get('cut_off_date', cut_off_date_input.isoformat())}` | "
                                f"**Year:** `{api_response.get('year', cut_off_date_input.year)}`")

                    histogram_df = process_histogram_prediction(api_response)
                    
                    if not histogram_df.empty:
                        # Display bar chart
                        st.bar_chart(histogram_df, x='Bin Range', y='Probability', height=500)
                        
                        # Display raw data in an expander (optional)
                        with st.expander("View Raw API Response"):
                            st.json(api_response)
                        with st.expander("View Processed Chart Data"):
                            st.dataframe(histogram_df)
                    else:
                        st.warning("Could not process histogram data from the API response. Ensure the response format is correct.")
                        with st.expander("View Raw API Response"):
                            st.json(api_response)
                
                elif api_response and api_response.get("error"):
                    st.error(f"API Error: {api_response.get('error')}")
                    with st.expander("View Full API Error Response"):
                        st.json(api_response)
                else:
                    st.error("Received an unexpected or empty response from the prediction API.")
                    with st.expander("View Raw API Response"):
                        st.json(api_response if api_response else "No response object received.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # Consider adding more specific error messages based on exception type
else:
    st.info("Select prediction inputs in the sidebar and click 'Get Prediction'.")
    # Optional: Show a placeholder image or example chart
    st.markdown("### Example Histogram Output")
    example_bins = [f"{i}-{i+20}" for i in range(0, 100, 20)]
    example_probs = [0.1, 0.25, 0.3, 0.25, 0.1]
    if len(example_bins) > len(example_probs): # Quick fix if lengths mismatch for placeholder
        example_bins = example_bins[:len(example_probs)]
    elif len(example_probs) > len(example_bins):
        example_probs = example_probs[:len(example_bins)]

    if example_bins and example_probs: # Ensure lists are not empty
        example_df = pd.DataFrame({
            'Bin Range': example_bins,
            'Probability': example_probs
        })
        st.bar_chart(example_df, x='Bin Range', y='Probability', height=400)
    else:
        st.markdown("_Placeholder chart could not be generated._")


# --- (Optional) Footer or additional information ---
st.markdown("---_Dashboard interacting with the AgriYield Prediction Service._---")
