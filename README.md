# AgriYield: A Cloud-Native MLOps System for Crop Yield Prediction
### Group 4

### Project Overview
**Value Proposition:** AgriYield aims to improve the accuracy of end-of-season crop yield predictions and provide **timely, data-driven forecast updates throughout the growing season** compared to traditional methods (heuristics, historical averages, manual analysis) used by agricultural planners, researchers, and related stakeholders. The system focuses on predicting **county-level yield variability** driven by weather and satellite-observed conditions, offering insights beyond static historical averages or lagged official reports. It leverages large-scale satellite imagery and weather data, primarily via the pre-processed CropNet dataset, combined with modern ML techniques within a robust MLOps framework.

**Status Quo:** Current methods often rely on USDA reports (which have lag), farmer surveys, historical county averages, or qualitative assessments. These are often slow to adapt to intra-season weather variations and emerging climate change impacts.

----

**Business Metric:**
1.  **Yield Prediction Accuracy:** Measured quantitatively by Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) in yield units (e.g., bushels/acre) compared to actual USDA NASS county yields on held-out data (years/regions). 
2.  **Timeliness of Forecasts:** Measured by the time taken to generate and deliver updated yield forecasts to users (e.g., agricultural departments, cooperatives) compared to traditional methods. This includes the speed of processing incoming data and generating predictions.
3.  **User Engagement:** Measured by the number of active users accessing the system for yield predictions and updates, indicating the system's adoption and utility in real-world scenarios.

----
**Target Users:** Regional agricultural departments, farming cooperatives, commodity traders, agricultural researchers, and policymakers. The system enhances existing planning and analysis workflows. Some example of potential users include:
* **USDA NASS:** For accurate county-level yield estimates to inform agricultural policy and commodity markets.
* **Farmers and Cooperatives:** For real-time yield forecasts to optimize planting, harvesting, and resource allocation.
* **Commodity Traders:** For data-driven insights into crop supply and demand dynamics, aiding in trading decisions.
* **Agricultural Researchers:** For studying the impact of weather and climate on crop yields, supporting research in agronomy and climate science.


### Contributors

| Name                   | Responsible for                                      | Link to their commits |
| :--------------------- | :--------------------------------------------------- | :----------------------------------------------- |
| All team members       | Overall system design, integration, final reporting  |                                                  |
| Saad Zubairi           | Model Training & Training Platforms (Units 4, 5)     |                                                  |
| Ali Hamza              | Model Serving & Monitoring Platforms (Units 6, 7)    |                                                  |
| Armish Javed           | Data Pipeline (Unit 8)                               |                                                  |
| Garwith Luhadia        | Continuous X / DevOps (Unit 3)                       |                                                  |

### System diagram

![System Diagram Placeholder](sys_diag.jpeg) 

The diagram illustrates the system architecture deployed on Chameleon Cloud.

### Summary of outside materials

| Material             | How it was created                                                                                                                                                           | Conditions of use                                                                                                                  | Size |
| :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |------ |
| **CropNet Dataset** | Created by academic researchers [https://arxiv.org/abs/2406.06081]. Combines Sentinel-2, WRF-HRRR weather, and USDA NASS yields for ~2.3k US counties (2017-2022). | Open dataset for research. https://github.com/fudong03/CropNet |1 TB
| (Implicit) Sentinel-2 | ESA Copernicus Program satellite imagery.                                                                                                                                    | Free and open data policy, requires attribution to Copernicus.                                                                    |
| (Implicit) WRF-HRRR | Derived from NOAA's HRRR weather model, processed for CropNet.                                                                                                               | Underlying NOAA data is public domain.                                                   |
| (Implicit) USDA NASS Yields | Official US county yield statistics collected via surveys by USDA NASS.                                                                                              | Public domain (US Government work).                                                                                                |

| (Optional) SSURGO/gSSURGO | Soil data from USDA NRCS.                                                                                                                                            | Public domain. May be used to supplement CropNet features if needed.                                                               |
The CropNet dataset is the primary data source for this project. It provides a comprehensive set of features for training and evaluating the yield prediction model. The other datasets are used implicitly as part of the CropNet dataset. We may also consider using SSURGO/gSSURGO soil data to enhance the model's performance, but this is a necessary step.

---- 

### Summary of infrastructure requirements

| Requirement          | How many/when                                                                 | Justification                                                                                                                                                                     |
| :------------------- | :---------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `m1.large` (or similar CPU VMs) | ~6-7 nodes allocated for entire project duration                             | K3s master (1), K3s workers (min 2-3 for services/jobs), Ray head/worker (min 2-3, can overlap w/ K8s), MLflow (1), Dashboard (1). Provides compute for cluster management and services. |
| `gpu_a100` / `gpu_mi100` (or available GPU type) | ~24-32 GPU-hours/week during training period | Needed for efficient training/tuning of CNN/Transformer models on CropNet data. Required for distributed training experiments (Unit 4 difficulty). Potentially for optimized inference later. |
| Floating IPs         | 1 for entire project duration                                                 | 1 for stable external access to K8s API / MLflow UI / Dashboard / API server.                                                                                                           |
| Persistent Storage   | ~2 TB block storage allocated for entire project duration    | Required for storing the large CropNet dataset (~1TB+), ML model artifacts/checkpoints, container images cache, logs, and intermediate data generated by pipelines.                      |

---

### Detailed design plan

#### Model training and training platforms

1.  *Strategy:* We will be implementing a multi-modal DL model (ResNet for Sentinel-2 image patches fused with a sequential transformer models for the WRF-HRRR weather sequences sourced from the CropNet dataset) using PyTorch. Special considerations will be given to approaches that use 2D CNNs or 3D CNNs that could effectively convolve accross space and time.
2.  *Infrastructure:* To define the multi-branch CNNs, we will use PyTorch; For scaling the model and parallelize training, we will be using Ray; We will be considering deploying kubernetes clusters on the baremetal reconfigurable hardware from Chameleon. For tracking experiments and versioning the models, we'll use MLFlow.
3.  *Justification:* Leverages state-of-the-art approaches from literature for yield prediction accuracy using multi-modal data. Ray enables required scaling. MLflow ensures reproducibility and traceability critical for MLOps.
4.  *Lecture Link:* Relates to Unit 4 (Modeling choices, Training at Scale, Distributed Training Strategies like DDP) and Unit 5 (Experiment Tracking with MLflow, Training Infrastructure using Ray).
5.  *Specifics:* R² (coefficient of determination) is reported to indicate the fraction of variance explained. For example, an R² of 0.85 at county level for corn is considered state-of-the-art. Some use MAPE (Mean Absolute Percentage Error) to normalize across crops or regions.
6.  *Unit Requirements Satisfied:* Unit 4 (Train/Retrain model, Modeling choices justified), Unit 5 (Experiment tracking hosted on Chameleon, Scheduling training jobs on Ray cluster).
7.  *Difficulty Points Attempted:* *Unit 4: Use distributed training to increase velocity.* (Plan to include plot of time vs. #GPUs in final report). Will implicitly use Ray Train (Unit 5 requirement) for its execution features like checkpointing.

#### Model serving and monitoring platforms

1.  *Strategy:* We deploy the trained model using FastAPI as a REST API, packaged with Docker, and run it on a Kubernetes cluster. After training, we evaluate the model using standard metrics and monitor its performance over time.
2.  *Diagram Parts:* FastAPI Model Service, Docker containers, CI/CD pipeline, Monitoring scripts, Model Registry, Dashboard for results.
3.  *Justification:* This setup is easy to scale, follows modern cloud practices, and helps keep the model up to date and reliable.
4.  *Lecture Link:* Covers Unit 6 (model deployment with API, containerization) and Unit 7 (basic evaluation and monitoring after deployment).
5.  *Specifics:* The model should respond within 2 seconds, support at least 10 users at once, and update predictions for all counties in under 1 hour. We check accuracy with basic metrics (RMSE, MAE, R²) and look for big changes in input data.
6.  *Unit Requirements Satisfied:* Unit 6: API for serving, Docker/K8s deployment; Unit 7: Offline evaluation, basic load testing, simple monitoring and feedback loop.
7.  *Difficulty Points Attempted:* *Unit 7: Monitor for data drift.* Added basic data drift detection and seasonal performance checks to catch problems early.

#### Data pipeline

1.  *Strategy:* We store the CropNet dataset on Chameleon’s persistent storage and access it during training and evaluation. We simulate online data access with a script that fetches features based on county and year. A simple dashboard will help us visualize the data and model predictions.
2.  *Diagram Parts:* Persistent Storage, Docker containers running training or data services, DataLoader code, Feature fetch script, Dashboard service.
3.  *Justification:* Handling a large dataset like CropNet requires fast and reliable storage and loading. The online simulation checks that the model can retrieve the right features, and the dashboard makes the data easier to understand and debug.
4.  *Lecture Link:* Covers Unit 8 (Storage on Chameleon, ETL/loading for training, simulating online data access).
5.  *Specifics:* Dataset is around 1TB. We load batches from storage during training. The online script returns features for a given county and date. The dashboard lets users explore inputs and compare predicted vs. actual crop yield. 
6.  *Unit Requirements Satisfied:* Unit 8: Persistent storage, offline data access with DataLoaders, online data simulation, basic ETL defined.
7.  *Difficulty Points Attempted:* *Unit 8: Interactive data dashboard.*

#### Continuous X

1.  *Strategy:* Infrastructure (VMs, storage) will be defined as code (Terraform/`python-chi` in Git). Application configurations (Kubernetes manifests, service settings) will also be stored in Git and automatically applied to the K3s cluster. CI (using GitHub Actions) will handle testing and building Docker images. CD/CT will automate the end-to-end process: triggering training on Ray, evaluating, registering models in MLflow, deploying the API server to staging, running tests, and simulating promotion through canary to production environments, adhering to immutable infrastructure principles.
2.  *Diagram Parts:* Git Repo (GitHub), CI/CD Tools (GitHub Actions), Container Registry, K8s Cluster (Staging/Canary/Prod namespaces), Ray Cluster, MLflow Server.
3.  *Justification & Lecture Link (Unit 3):* This implements a fully automated, reproducible MLOps workflow using IaC, GitOps, CI/CD/CT pipelines, containerization, and staged deployments, directly applying the core DevOps and cloud-native principles from Unit 3.
4.  *Specifics & Requirements:* Terraform/python-chi defines infra. K8s YAML defines apps. This satisfies **Unit 3** (IaC, Cloud-native principles, CI/CD/CT pipeline, Staged deployment).
7.  *Difficulty Points Attempted:* *Unit 1: Composed of multiple models.* (The system design inherently supports multiple interacting models - yield predictor, potentially recommender, integrated and managed via this Continuous X pipeline).