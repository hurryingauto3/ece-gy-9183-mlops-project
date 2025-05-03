terraform {
  required_version = ">= 0.14.0"
  required_providers {
    openstack = {
      source = "terraform-provider-openstack/openstack"
      version = "~> 1.51.1"  # use a recent version
    }
  }
}

provider "openstack" {
  cloud = var.openstack_cloud  # Name of the OpenStack cloud in clouds.yaml (e.g., "CHI-UC")
}
