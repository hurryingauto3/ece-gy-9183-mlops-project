To start model training on bare-metal GPU server:

- Start bare-metal GPU server with Terraform

```bash
cd chameleon_devops/terraform
terraform init
terraform apply -auto-approve
```

- ssh and install docker and nvidia toolkit I think, if not already there

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@<GPU_IP>

# On the GPU server:
sudo apt update && sudo apt install -y curl git

# this is probably optional because the ubuntu images used already have the following installed but just incase:
------------------------------------
# Install Docker
curl -sSL https://get.docker.com/ | sudo sh
sudo groupadd -f docker
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
------------------------------------

# Verify GPU visibility:
docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi
```


