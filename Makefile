train:
	python model_training/train.py --config configs/model.yaml

train-ray:
	bash devops/scripts/train_ray.sh

serve-api:
	docker build -t agri-api model_serving/
	docker run -p 8000:8000 agri-api

deploy-api:
	bash devops/scripts/deploy_api.sh

dashboard:
	streamlit run data_pipeline/dashboard.py