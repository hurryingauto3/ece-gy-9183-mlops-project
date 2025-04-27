terraform {
  required_providers {
    openstack = {
      source = "terraform-provider-openstack/openstack"
      version = "1.49.0"  # use a recent version
    }
  }
}

provider "openstack" {
  cloud = var.openstack_cloud  # Name of the OpenStack cloud in clouds.yaml (e.g., "CHI-UC")
}
