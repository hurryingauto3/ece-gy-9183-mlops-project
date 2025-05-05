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

- **site.yml** orchestrates:
  - `docker_install` & `docker_compose` (local Compose based default)
  - `project` & `services` (clone repo, build images, start compose)
  - `gpu_setup` (install NVIDIA runtime on GPU node)
  - `k3s` (install K3s on service node)
  - `argocd` (deploy ArgoCD via Helm)
  - `argocd_image_updater` (sync env overlays & image updates)
  - `workflows` (install Argo Workflows CRDs, controllers, CronWorkflows)
  - `monitoring` (deploy kube‑prometheus‑stack)
  - `argo_workflows` (optional Argo Workflows standalone Helm chart)

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

- **Kustomize overlays** under `k8s/overlays/{staging,canary,production}`  
- **ArgoCD ApplicationSet** in `k8s/argocd/applicationset.yaml` auto‑syncs all envs  
- **Image‑Updater hooks** pick semver‑prefixed tags (`staging-*, canary-*, prod-*`) and commit back on bump  

---

## Congratulations!

Your MLOps platform is now live!  
- **MLflow** @ `http://<services-ip>:5000`  
- **Grafana** @ `http://<services-ip>:3000`  
- **Prometheus** @ `http://<services-ip>:9090`  
- **ArgoCD** @ `http://<services-ip>:32443`  
- **Argo Workflows** UI via ArgoCD or `kubectl port-forward svc/argo-workflows-server -n argo 2746:2746`

Feel free to explore, customize, and contribute!  
