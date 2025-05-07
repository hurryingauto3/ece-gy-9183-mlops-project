# 🌾 AgriYield Data Processing Instructions

### Overview

- **Geographic unit**: `County` (using shapefiles)
- **Time unit**: `5-day intervals` — 73 points per year
- **Time span**: `2017–2022` (6 years total)

---

## ✅ Step-by-Step Pipeline

### Mask Cropland Only
- Use the **USDA Cropland Data Layer (CDL)** for each year.
- Exclude all pixels that are not cropland (e.g., forest, buildings, water).
- Resample CDL (30m) to match Sentinel-2 resolution (10m) using **nearest-neighbor**.
- Apply this mask to each Sentinel-2 image.

---

### Group Data by County
- Use **county shapefiles** to clip both Sentinel-2 imagery and WRF-HRRR weather data.
- Organize the data **per county and per year**.
- This forms one dataset unit per **county-year**.

---

### Compute NDVI Time Series
- For each Sentinel-2 image (every ~5 days), calculate:
  \[
  \text{NDVI} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}
  \]
- Use only **masked cropland pixels**.
- Aggregate the **mean NDVI** per county for each 5-day interval.
- Final result: a **(73,)** NDVI vector for every county-year.

---

### Integrate Weather Data
- Extract WRF-HRRR weather features (e.g., temperature, precipitation, humidity).
- Aggregate them per county and match to the same 5-day periods.
- Combine with NDVI to get feature vectors of shape **(73, num_features)** per sample.

---

### Attach Yield Labels
- For each **county-year**, attach the scalar label: **bushels per acre**.
- This becomes the `y` value for ML.

---

### Build Final Dataset
- `X` shape: **(num_samples, 73, num_features)**  
  Each sample = one county-year (e.g., "Fresno, CA – 2019")
- `y` shape: **(num_samples,)**  
  Scalar yield values per county-year.

---

### Sanity Checks
- Plot NDVI curves for a few counties
- Check for flat or missing NDVI (due to clouds or bad tiles)
- Ensure 73 timepoints exist for each county-year
- Confirm total samples ≈ 2291 counties × 6 years (minus any missing)

---

### ✅ Summary

Each training sample = 1 county × 1 year  
Each sample contains:
- NDVI + weather over 73 timepoints
- A yield label (bushels/acre)

Format is ready for time series regression in PyTorch or other ML frameworks.
