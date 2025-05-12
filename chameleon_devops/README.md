# Chameleon MLOps DevOps

A complete Infrastructure-as-Code and GitOps setup to provision, configure, and deploy an end-to-end MLOps stack on TACC's Chameleon testbeds using Terraform, Ansible, Docker Compose, and ArgoCD.

---

## Quick Start

1. **Set your GPU lease**  
   Copy your Chameleon GPU reservation UUID into `chameleon_devops/terraform/terraform.tfvars` under `gpu_reservation_id`.

2. **Provision Infrastructure**  
   ```bash
   cd chameleon_devops/terraform
   terraform fmt
   terraform validate
   terraform plan -var-file=terraform.tfvars
   terraform apply -auto-approve -var-file=terraform.tfvars
   terraform output -raw ansible_inventory > ../ansible/inventory.ini
   ```

3. **Configure & Deploy with Ansible**  
   ```bash
   cd chameleon_devops/ansible
   ansible-playbook -i inventory.ini site.yml --ask-vault-pass
   ```

---

## Prerequisites

- **Terraform v1.0+**  
- **Ansible v2.12+** with collections: `kubernetes.core`, `community.general`  
- **SSH key** configured in `~/.ssh` and referenced by `terraform.tfvars`  
- **Vault password** for Ansible secrets  
- **Kubeconfig** permissions on control node  

---

## Terraform Workflow

1. **Define variables** in `terraform.tfvars` (lease ID, network, images, flavors).  
2. **Format & Validate**  
   ```bash
   terraform fmt && terraform validate
   ```  
3. **Plan & Apply**  
   ```bash
   terraform plan -var-file=terraform.tfvars
   terraform apply -auto-approve -var-file=terraform.tfvars
   ```  
4. **Extract Inventory**  
   ```bash
   terraform output -raw ansible_inventory > ../ansible/inventory.ini
   ```

---

## Ansible Workflow

The root playbook [`chameleon_devops/ansible/site.yml`](chameleon_devops/ansible/site.yml) orchestrates everything in discrete “plays”:

1. **Shared Base Provisioning** (`hosts: all`)  
   - **Role:** `docker_install`  
     - Installs Docker Engine on every node  
     - [`chameleon_devops/ansible/roles/docker_install/`](chameleon_devops/ansible/roles/docker_install/)  
   - **Role:** `docker_compose`  
     - Installs Docker Compose plugin  
     - [`chameleon_devops/ansible/roles/docker_compose/`](chameleon_devops/ansible/roles/docker_compose/)  
   - **Role:** `copy_configs`  
     - Copies Docker‐Compose YAMLs, Prometheus configs, Argo manifests into the project directory  
     - [`chameleon_devops/ansible/roles/copy_configs/`](chameleon_devops/ansible/roles/copy_configs/)  
   - **Role:** `project`  
     - Clones `ece-gy-9183-mlops-project`, checks out the right branch, ensures directory structure  
     - [`chameleon_devops/ansible/roles/project/`](chameleon_devops/ansible/roles/project/)  

2. **GPU Node Setup** (`hosts: gpu_nodes`)  
   - **Role:** `gpu_setup`  
     - Installs NVIDIA drivers & configures the Docker NVIDIA runtime  
     - [`chameleon_devops/ansible/roles/gpu_setup/`](chameleon_devops/ansible/roles/gpu_setup/)  
   - **Task:**  
     ```yaml
     shell: docker compose --profile gpu up -d --remove-orphans
     args:
       chdir: "{{ project_remote_path }}"
     ```
     - Spins up GPU‐specific containers (e.g. training environment)

3. **Services Node Setup** (`hosts: services_nodes`)  
   - **Task:** Ensure Prometheus config directory & file exist under `devops/docker-compose`  
   - **Task:**  
     ```yaml
     shell: docker compose --profile services up -d --remove-orphans
     args:
       chdir: "{{ project_remote_path }}"
     ```
     - Brings up MLflow, FastAPI, Ray Dashboard, MinIO, Grafana, Prometheus, Locust, etc.

4. **Persistent Storage Migration** (`hosts: services_nodes,gpu_nodes`)  
   - **Tasks:**  
     - Detect and format `/dev/vdb`  
     - Mount it at `/mnt/persistent`  
     - Rsync `/var/lib/docker` → `/mnt/persistent/docker`  
     - Write `/etc/docker/daemon.json` to point `data-root` to the new location  
     - Restart Docker

5. **Kubernetes Control‑Plane Setup** (`hosts: k8s_control_plane`)  
   - **Role:** `k3s`  
     - Installs K3s server, joins workers  
     - [`chameleon_devops/ansible/roles/k3s/`](chameleon_devops/ansible/roles/k3s/)  
   - **Role:** `argocd`  
     - Deploys Argo CD in `argocd` namespace via Helm  
     - [`chameleon_devops/ansible/roles/argocd/`](chameleon_devops/ansible/roles/argocd/)  
   - **Role:** `argocd_image_updater`  
     - Configures the Image Updater to watch semver tags & sync Kustomize overlays  
     - [`chameleon_devops/ansible/roles/argocd_image_updater/`](chameleon_devops/ansible/roles/argocd_image_updater/)  
   - **Role:** `workflows`  
     - Installs Argo Workflows CRDs, controller, CronWorkflows  
     - [`chameleon_devops/ansible/roles/workflows/`](chameleon_devops/ansible/roles/workflows/)  
   - **Role:** `monitoring`  
     - Deploys the kube‑prometheus‑stack (Prometheus + Grafana)  
     - [`chameleon_devops/ansible/roles/monitoring/`](chameleon_devops/ansible/roles/monitoring/)  

> **Note:** Secrets (GHCR PAT, MinIO keys, Vault credentials) are pulled in via  
> `chameleon_devops/ansible/group_vars/all/secrets.yml` (encrypted with Ansible Vault).  
>  
> This single `site.yml` playbook lets us go from bare nodes to a fully‑functional MLOps platform with zero manual steps.


---

## Secrets & Vault

- **`ansible/vault`**  
  Use `ansible-vault encrypt group_vars/all/secrets.yml` to store:
  ```yaml
  ghcr_username: YOUR_GHCR_USER
  ghcr_pat: YOUR_GHCR_PAT
  minio_access_key: admin
  minio_secret_key: password
  ```
- **Load secrets** via `vars_files` in `site.yml`.

---

## Environments & GitOps

- **Raw Manifests** in `k8s/`  
  - Core Kubernetes resources live under:  
    - `k8s/model_serving/deployment.yaml` & `service.yaml`  
    - `k8s/argo/argo-cd.yaml` (ApplicationSet + ArgoCD server config)

- **Kustomize “base” + overlays**  
  - **Base** manifests (`k8s/model_serving/*`, `k8s/argo/*`) define the core Deployment, Service, and ArgoCD ApplicationSet.  
  - **Overlays** under `k8s/overlays/{staging,canary,production}` each include a `kustomization.yaml` that:  
    - References the base manifests  
    - Applies environment‐specific patches (replica counts, resource limits, namespace, image tag overrides)  
    - Lives at:  
      - `k8s/overlays/staging/kustomization.yaml`  
      - `k8s/overlays/canary/kustomization.yaml`  
      - `k8s/overlays/production/kustomization.yaml`

- **ArgoCD ApplicationSet** (`k8s/argocd/applicationset.yaml`)  
  - Uses the **List** generator to spawn one Application per environment, pointing to `k8s/overlays/{{path}}` in Git.  
  - Example snippet:
    ```yaml
    generators:
      - list:
          elements:
            - name: staging
              path: overlays/staging
            - name: canary
              path: overlays/canary
            - name: production
              path: overlays/production
    template:
      metadata:
        name: '{{name}}-app'
      spec:
        source:
          repoURL: 'https://github.com/hurryingauto3/ece-gy-9183-mlops-project.git'
          path: 'k8s/overlays/{{path}}'
          targetRevision: main
        destination:
          server: 'https://kubernetes.default.svc'
          namespace: '{{name}}'
    ```

- **ArgoCD Image Updater**  
  - Installed via the Ansible role `argocd_image_updater` in `chameleon_devops/ansible/roles/argocd_image_updater/`.  
  - Deploys an `ImageUpdateAutomation` CRD in the ArgoCD namespace that:  
    - Watches for new image tags prefixed by `staging-`, `canary-`, and `prod-`  
    - Automatically patches the corresponding overlay’s `kustomization.yaml` with the updated tag  
    - Commits the change back to the Git repo, triggering a fresh sync by ArgoCD

- **GitOps Workflow**  
  1. **CI** builds & tags a new container image (`staging-<version>`) and pushes to GHCR/ECR.  
  2. **Image Updater** sees the new `staging-<version>` tag, updates `k8s/overlays/staging/kustomization.yaml`, commits to Git.  
  3. **ArgoCD ApplicationSet** detects the Git change and auto‑syncs the `staging` namespace.  
  4. Once `staging` passes smoke tests, CI or manual step tags the image `canary-<version>`, repeating the update/sync for `canary`.  
  5. After load‑testing in `canary`, a `prod-<version>` tag triggers the final rollout to `production`.

- **Secret & Access Management**  
  - ArgoCD uses an SSH deploy key (created by `argocd` Ansible role) to pull manifests from the private Git repo.  
  - Overlays can include environment‑specific secrets via sealed‑secrets or external vault integrations.

This setup ensures zero‑touch promotion from `staging` → `canary` → `production`, with all changes tracked via Git and reconciled automatically by ArgoCD.
 

---

## Platform Live & Service Endpoints

Your MLOps platform is now live! Access the following services:

| Service                  | URL / Access Method                                                                                  |
|--------------------------|------------------------------------------------------------------------------------------------------|
| **MLflow**               | `http://<services-ip>:5000` – experiment tracking & model registry                                   |
| **Grafana**              | `http://<services-ip>:3000` – system & model performance dashboards                                  |
| **Prometheus**           | `http://<services-ip>:9090` – metrics scrape endpoint                                                |
| **MinIO**                | `http://<services-ip>:9000` – S3‑compatible object store (datasets & artifacts)                      |
| **MinIO Console**        | `http://<services-ip>:9001` – web UI for bucket & object management _(default: `admin` / `password`)_ |
| **FastAPI Inference**    | `http://<services-ip>:8000` – online inference endpoint                                              |
| **Ray Dashboard**        | `http://<services-ip>:8265` – monitor distributed training & Ray cluster                              |
| **Ray Client**           | Connect via `python -m ray <…> --address='ray://<services-ip>:10001'`                                 |
| **DataViz Dashboard**    | `http://<services-ip>:8501` – Streamlit/Gradio UI for interactive prediction exploration              |
| **Argo CD (HTTP)**       | `http://<services-ip>:80` – GitOps web interface                                                     |
| **Argo CD (HTTPS/gRPC)** | `https://<services-ip>:443` – secure API for manifest sync & Image Updater                           |                             |                |
| **Locust Load Testing**  | `http://<services-ip>:8089` – Locust swarm master for performance & regression testing                |

> **Default MinIO credentials:**  
> `Access Key: admin`  
> `Secret Key: password`

_Ensure your firewall/security groups allow these ports on the services node._


---

## Congratulations!

Your MLOps platform is now live!  

Feel free to explore, customize, and contribute!  
