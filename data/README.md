# HRRR Crop Yield Data Lake

This repository organizes weather time-series data and USDA crop yield targets for multiple crops and counties, designed for machine learning applications in crop yield prediction.

## Folder Structure

- `[FIPS_CODE]`: 5-digit county FIPS code (e.g., `19109` for Kossuth County, Iowa).
- `[Crop_Type]`: Name of the crop (e.g., `Corn`, `Soy`, `Cotton`, `WinterWheat`).
- `WeatherTimeSeries[Year].csv`: Weather features for the specified year and crop.
- `USDACropyield.json`: Mapping from year to USDA-reported crop yield for that county and crop.

---

## 📦 File Details

### `WeatherTimeSeries[Year].csv`
- Contains daily (or sub-daily) weather observations for the entire year.
- Typical columns include:
  - `date`
  - `temperature`
  - `precipitation`
  - `wind_speed`
  - `solar_radiation`
  - Other meteorological features

### `USDACropyield.json`
- A simple JSON mapping from year to yield value.
- Example:
  ```json
  {
    "2020": 175.3,
    "2021": 168.7,
    "2022": 182.1
  }