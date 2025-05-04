terraform {
  required_version = ">= 0.14.0"
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.51.1"
    }
  }
}

provider "openstack" {
  alias  = "kvm"
  cloud  = var.cloud           # "kvm"
  region = var.services_region # "KVM@TACC"
}

provider "openstack" {
  alias  = "chi"
  cloud  = "chi"          # name of your clouds.yaml entry
  region = var.gpu_region # "CHI@TACC"
}

provider "openstack" {
  alias  = "uc"
  cloud  = var.cloud_uc
  region = var.uc_region
}