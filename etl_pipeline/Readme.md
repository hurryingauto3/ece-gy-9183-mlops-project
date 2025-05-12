name: MLOps ETL Pipeline - Extract, Transform, Load
description: >
  This project implements a complete Extract-Transform-Load (ETL) pipeline for preparing agricultural machine learning data.
  It runs on a GPU node with persistent storage mounted at /mnt/swift_store and processes HRRR weather and USDA yield data.

sections:
  - title: Directory Structure
    
      ```
      etl_pipeline/
      ├── extract/
      │   ├── extract.py           # Downloads & cleans raw HRRR/USDA data
      │   └── Dockerfile           # Container to run the extract job
      ├── transform/
      │   ├── transform.py         # Parses, merges, and outputs structured datasets
      │   └── Dockerfile           # Container to run the transform/load job
      ├── docker-compose.yml       # Defines extract and transform services
      └── /mnt/swift_store/        # Mounted persistent volume (on GPU node)
          ├── raw_data/            # Output from extract
          └── transformed_data/    # Output from transform/load
      ```

  - title: What the Pipeline Does
    
    
      ### Extract Phase
      - Downloads HRRR and USDA datasets using `gdown` from Google Drive.
      - Filters out unwanted years (keeps 2017–2021).
      - Stores files to `/mnt/swift_store/raw_data/`.

      ### Transform Phase
      - Aggregates HRRR weather data by FIPS and date.
      - Dynamically identifies and merges USDA yield data by FIPS/year.
      - Keeps everything in memory for efficiency.

      ### Load Phase
      - Saves clean output datasets to `/mnt/swift_store/transformed_data/`:
        - `training.csv` — for model training (2017–2020)
        - `test.csv` — for evaluation (2021 even days)
        - `production/` — split `val` (2021 odd days) into:
          - `dev.csv`
          - `staging.csv`
          - `canary.csv`

  - title: Environment Setup (GPU Node)
    
      ### 1. SSH into the GPU node
      ```bash
      ssh -i ~/.ssh/mlops_proj_key cc@A.B.C.D
      ```

      ### 2. Create a `.env` file
      ```bash
      cd ~/etl_pipeline
      nano .env
      ```

      Add folder IDs:
      ```env
      GOOGLE_DRIVE_FOLDER_ID_HRRR=your_hrrr_folder_id
      GOOGLE_DRIVE_FOLDER_ID_USDA=your_usda_folder_id
      ```

  - title: Running the Pipeline (Docker Compose)
    
      ### Build Services
      ```bash
      docker compose build etl-download etl-transform
      ```

      ### Run Extract Phase
      ```bash
      docker compose run etl-download
      ```

      ### Run Transform + Load Phase
      ```bash
      docker compose run etl-transform
      ```

  - title: Move Production Data to Services Node
    
      From the GPU node, run the following to transfer production files:
      ```bash
      scp -i ~/.ssh/mlops_proj_key /mnt/swift_store/transformed_data/production/*.csv \
        cc@<SERVICES_NODE_IP>:/mnt/swift_store/env_split/val_2022/
      ```

      Replace `<SERVICES_NODE_IP>` with the IP address of  services VM.

  
  - title:  Delete Production Data from GPU After Verification
    
      After verifying the files are present and correct on the services node,
      remove the production data from the GPU node to prevent duplication and save space:

      ```bash
      rm -rf /mnt/swift_store/transformed_data/production
      ```

  - title: Outputs
    
      After successful execution:
      ```
      /mnt/swift_store/
      ├── raw_data/
      │   ├── WRF-HRRR Computed dataset/
      │   └── USDA Crop Dataset/
      └── transformed_data/
          ├── training.csv
          ├── test.csv
      ```

  - title: Requirements
    
      - Python 3.8+
      - `gdown`, `pandas`
      - Docker + Docker Compose
      - Mounted volume at `/mnt/swift_store`

  - title: Summary
    
      | Step     | Input                      | Output                                  |
      |----------|-----------------------------|-----------------------------------------|
      | Extract  | Google Drive folders        | `/mnt/swift_store/raw_data/`           |
      | Transform| Raw HRRR & USDA data        | In-memory merge                         |
      | Load     | Transformed merged dataset  | `training.csv`, `test.csv`, prod splits |