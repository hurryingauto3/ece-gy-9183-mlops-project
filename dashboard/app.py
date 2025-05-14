import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date

# Import new and existing services/processors
from api_service import fetch_single_crop_histogram_prediction
from data_processor import process_histogram_prediction

# Prometheus client
# Import REGISTRY to access the default registry
from prometheus_client import Counter, Gauge, start_http_server, REGISTRY, CollectorRegistry
import threading
import atexit

# --- Prometheus Metrics Setup ---

PROMETHEUS_PORT = 9095 # Internal port for Prometheus scraping within Docker network

def initialize_metrics_and_server():
    # Use a session state variable as a flag
    if 'prometheus_setup_complete' not in st.session_state:
        print("INFO: Performing Prometheus metrics and server setup...")

        try:
            # Attempt to get existing metrics
            # If they don't exist, we'll hit a KeyError or similar, and then create them.
            page_views = REGISTRY._names_to_collectors['dashboard_page_views']
            prediction_requests = REGISTRY._names_to_collectors['dashboard_prediction_requests_total']
            active_sessions = REGISTRY._names_to_collectors['dashboard_active_sessions']
            last_prediction_time = REGISTRY._names_to_collectors['dashboard_last_prediction_timestamp_seconds']
            print("INFO: Prometheus metrics found in registry.")

        except (KeyError, AttributeError):
            # Metrics don't exist in the registry yet, create them
            print("INFO: Creating Prometheus metrics and registering with default registry.")
            page_views = Counter('dashboard_page_views', 'Dashboard page view count')
            prediction_requests = Counter('dashboard_prediction_requests_total', 'Total prediction requests made from dashboard')
            active_sessions = Gauge('dashboard_active_sessions', 'Number of active dashboard user sessions')
            last_prediction_time = Gauge('dashboard_last_prediction_timestamp_seconds', 'Timestamp of the last prediction request')

        # Store the metric objects in session state so they are accessible on subsequent reruns
        st.session_state.metrics = {
            'PAGE_VIEWS': page_views,
            'PREDICTION_REQUESTS': prediction_requests,
            'ACTIVE_SESSIONS': active_sessions,
            'LAST_PREDICTION_TIME': last_prediction_time,
        }

        # --- Start Prometheus HTTP Server (This part must run ONCE per process) ---
        # Use a global flag managed across all Streamlit reruns for the server
        # Streamlit's threading might still cause "Address already in use" if not careful.
        if 'prometheus_server_thread' not in st.session_state:
             try:
                 # Using threading.Thread ensures it doesn't block Streamlit
                 thread = threading.Thread(target=lambda: start_http_server(PROMETHEUS_PORT), daemon=True)
                 thread.start()
                 st.session_state.prometheus_server_thread = thread # Store the thread object
                 print(f"INFO: Prometheus metrics server starting in background thread on port {PROMETHEUS_PORT}")
             except Exception as e:
                 print(f"ERROR: Failed to start Prometheus metrics server thread: {e}")
        else:
             if not st.session_state.prometheus_server_thread.is_alive():
                  print("WARNING: Prometheus server thread died. Consider restarting the app.")
                  
        # Mark setup as complete for this session
        st.session_state.prometheus_setup_complete = True
        print("INFO: Prometheus setup marked as complete for this session.")
    else:
        print("INFO: Prometheus setup already complete for this session.")
        # Metrics should be accessible via st.session_state.metrics or REGISTRY._names_to_collectors

# Call the initialization function on every rerun
initialize_metrics_and_server()


PAGE_VIEWS = st.session_state.metrics['PAGE_VIEWS']
PREDICTION_REQUESTS = st.session_state.metrics['PREDICTION_REQUESTS']
ACTIVE_SESSIONS = st.session_state.metrics['ACTIVE_SESSIONS']
LAST_PREDICTION_TIME = st.session_state.metrics['LAST_PREDICTION_TIME']


# --- Application Logic ---

# Increment page view counter using the retrieved metric object
PAGE_VIEWS.inc()


# Page configuration (keep this part)
st.set_page_config(
    page_title="Crop Yield Histogram Prediction Dashboard",
    page_icon="🌾",
    layout="wide"
)

# Set title and description (keep this part)
st.title("🌾 Crop Yield Histogram Prediction")
st.markdown("""
    Select a FIPS county code, cut-off date, and crop type to visualize the predicted yield histogram.
""")

# Sidebar for input controls (keep this part)
st.sidebar.header("Prediction Inputs")

# FIPS County Code input (keep this part)
county_fips_input = st.sidebar.text_input(
    "FIPS County Code",
    value="DUMMY"
)

# Cut-off Date input (keep this part)
cut_off_date_input = st.sidebar.date_input(
    "Cut-off Date",
    value=date(datetime.now().year, 7, 15)
)

# Crop Type selection (keep this part)
crop_options = ["corn", "soybeans", "wheat", "rice", "cotton", "barley"]
selected_crop = st.sidebar.selectbox(
    "Select Crop Type",
    crop_options,
    index=0
)

# Histogram Bins (keep this part)
histogram_bins_str = st.sidebar.text_input(
    "Histogram Bin Edges (comma-separated)",
    value="0, 50, 100, 150, 200, 250",
    help="Enter comma-separated numbers for bin edges, e.g., 0,20,40,60,80,100. Must be at least 2 edges, strictly increasing."
)
st.sidebar.markdown("---_Note: The number of bins implied (edges - 1) must match the model's output._---")


# --- Main app layout --- (keep this part)
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
        # Use the metric objects retrieved from session state
        PREDICTION_REQUESTS.inc()
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
    # Optional: Show a placeholder image or example chart (keep this part)
    st.markdown("### Example Histogram Output")
    example_bins = [f"{i}-{i+20}" for i in range(0, 100, 20)]
    example_probs = [0.1, 0.25, 0.3, 0.25, 0.1]
    if len(example_bins) > len(example_probs):
        example_bins = example_bins[:len(example_probs)]
    elif len(example_probs) > len(example_bins):
        example_probs = example_probs[:len(example_bins)]

    if example_bins and example_probs:
        example_df = pd.DataFrame({
            'Bin Range': example_bins,
            'Probability': example_probs
        })
        st.bar_chart(example_df, x='Bin Range', y='Probability', height=400)
    else:
        st.markdown("_Placeholder chart could not be generated._")


# --- (Optional) Footer or additional information --- (keep this part)
st.markdown("---_Dashboard interacting with the AgriYield Prediction Service._---")