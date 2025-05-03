# terraform {
#   required_version = ">= 0.14.0"
#   required_providers {
#     openstack = {
#       source = "terraform-provider-openstack/openstack"
#       version = "~> 1.51.1"  # use a recent version
#     }
#   }
# }

# provider "openstack" {
#   cloud  = var.cloud           # Name of the OpenStack cloud from your clouds.yaml or RC
#   region = var.services_region # Region for Services node (e.g., "KVM@TACC")
# }

# provider "openstack" {
#   alias  = "gpu"
#   cloud  = var.cloud      # Same cloud but different region for GPU
#   region = var.gpu_region # Region for GPU node (e.g., "CHI@TACC")
# }

# provider "openstack" {
#   cloud  = var.cloud           # Name of the OpenStack cloud from your clouds.yaml or RC
#   region = var.services_region # Region for Services node (e.g., "CHI@TACC")
# }

# provider "openstack" {
#   alias  = "gpu"
#   cloud  = var.cloud      # Same cloud but different region for GPU
#   region = var.gpu_region # Region for GPU node (e.g., "CHI@UC")
# }

