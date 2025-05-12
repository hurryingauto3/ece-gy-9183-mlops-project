cd model_training

# Build image
```
docker build -f Dockerfile.jupyter -t agri-jupyter .
```

# Run with env vars loaded
```
set -a
. .env.jupyter
set +a
```

```bash
docker run --rm -it \
  --gpus all \
  -v $PWD:/app \
  -p 8888:8888 \
  --env-file .env.jupyter \
  agri-jupyter
```

Then tunnel to port 8888 from your local machine:
```
ssh -L 8888:localhost:8888 cc@<FLOATING_IP>
```

And open:
```
http://localhost:8888
```