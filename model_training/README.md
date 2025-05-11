
### TODO

>``export OS_AUTH_URL=… ``

>``export OS_APPLICATION_CREDENTIAL_ID=…``

>``export OS_APPLICATION_CREDENTIAL_SECRET=…``

>``export OS_SWIFT_CONTAINER_NAME=…``

fetch data:
> python fetch_data.py --fips 19001 --crop corn --out output

train and eval:

```
python train.py \
    --train-csv output/19001_corn_training_data.csv \
    --test-csv  output/19001_corn_testing_data.csv \
    --batch-size 32 \
    --epochs 20 \
    --mlflow
```

For train:

```bash
docker compose run model-training \
  python model_training/train.py \
    --train-csv model_training/input/12345_corn_training_data.csv \
    --test-csv  model_training/input/12345_corn_testing_data.csv \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --mlflow
```
infer:
```
docker compose run model-training \
  python model_training/inference.py \
    --input-csv model_training/input/your_partial_season.csv \
    --fips-id 0 \
    --num-samples 500
```
