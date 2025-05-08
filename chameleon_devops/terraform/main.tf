terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = ">= 1.0.0"
    }
  }
}

#############################################
# 1) OpenStack Providers
#############################################
provider "openstack" {
  alias  = "kvm"
  cloud  = var.cloud_kvm
  region = var.services_region
}

provider "openstack" {
  alias  = "chi"
  cloud  = var.cloud_chi
  region = var.chi_region
}

provider "openstack" {
  alias  = "uc"
  cloud  = var.cloud_uc
  region = var.uc_region
}

#############################################
# 2) External (public) network data sources
#############################################
data "openstack_networking_network_v2" "external_kvm" {
  provider = openstack.kvm
  name     = var.ext_net_name
}

data "openstack_networking_network_v2" "external_chi" {
  provider = openstack.chi
  name     = var.ext_net_name
}

#############################################
# 3) SSH Keypair import (KVM@TACC)
#############################################
resource "openstack_compute_keypair_v2" "keypair" {
  provider   = openstack.kvm
  name       = var.keypair_name
  public_key = file(var.public_key_path)
}

#############################################
# 4) Security Group + rules (KVM@TACC)
#############################################
resource "openstack_networking_secgroup_v2" "mlops_secgrp" {
  provider    = openstack.kvm
  name        = "${var.security_group_name}-${var.user_tag}"
  description = "Security group for MLOps VMs (${var.user_tag})"
}

resource "openstack_networking_secgroup_rule_v2" "ssh_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# (repeat for icmp_ingress, internal, mlflow_ui_ingress, grafana_ingress, etc.)

#############################################
# 5) Networking on KVM@TACC
#############################################
resource "openstack_networking_network_v2" "private_net_kvm" {
  provider       = openstack.kvm
  name           = "${var.network_name}-${var.user_tag}"
  admin_state_up = true
}

resource "openstack_networking_subnet_v2" "private_subnet_kvm" {
  provider   = openstack.kvm
  name       = "${var.network_name}-${var.user_tag}-subnet"
  network_id = openstack_networking_network_v2.private_net_kvm.id
  cidr       = var.network_cidr
  ip_version = 4
  gateway_ip = var.network_gateway
  allocation_pool {
    start = var.network_pool_start
    end   = var.network_pool_end
  }
  dns_nameservers = var.dns_nameservers
}

resource "openstack_networking_router_v2" "router_kvm" {
  provider            = openstack.kvm
  name                = "${var.network_name}-${var.user_tag}-router"
  admin_state_up      = true
  external_network_id = data.openstack_networking_network_v2.external_kvm.id
}

resource "openstack_networking_router_interface_v2" "router_intf_kvm" {
  provider  = openstack.kvm
  router_id = openstack_networking_router_v2.router_kvm.id
  subnet_id = openstack_networking_subnet_v2.private_subnet_kvm.id
}

#############################################
# 6) Block storage & attach (KVM@TACC)
#############################################
resource "openstack_blockstorage_volume_v3" "services_data" {
  provider    = openstack.kvm
  size        = 10
  name        = "project4-services-block-${var.user_tag}"
  description = "Block storage for Docker data on services-node"
}

resource "openstack_compute_volume_attach_v2" "attach_services_data" {
  provider    = openstack.kvm
  instance_id = openstack_compute_instance_v2.services_node.id
  volume_id   = openstack_blockstorage_volume_v3.services_data.id
  device      = "/dev/vdb"
}

#############################################
# 7) Services VM on KVM@TACC
#############################################
resource "openstack_compute_instance_v2" "services_node" {
  provider    = openstack.kvm
  name        = "services-node-project4-${var.user_tag}"
  image_name  = var.services_image
  flavor_name = var.services_flavor
  key_pair    = openstack_compute_keypair_v2.keypair.name

  security_groups = [
    openstack_networking_secgroup_v2.mlops_secgrp.id
  ]

  network {
    uuid = openstack_networking_network_v2.private_net_kvm.id
  }
}

#############################################
# 8) Private net on CHI@TACC (GPU & staging)
#############################################
resource "openstack_networking_network_v2" "private_net_chi" {
  provider              = openstack.chi
  name                  = "${var.network_name}-${var.user_tag}-chi"
  admin_state_up        = true
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet_chi" {
  provider   = openstack.chi
  name       = "${var.network_name}-${var.user_tag}-chi-subnet"
  network_id = openstack_networking_network_v2.private_net_chi.id
  cidr       = var.network_cidr
  ip_version = 4
  gateway_ip = var.network_gateway
  allocation_pool {
    start = var.network_pool_start
    end   = var.network_pool_end
  }
  dns_nameservers = var.dns_nameservers
}

resource "openstack_networking_router_v2" "router_chi" {
  provider            = openstack.chi
  name                = "${var.network_name}-${var.user_tag}-router-chi"
  admin_state_up      = true
  external_network_id = data.openstack_networking_network_v2.external_chi.id
}

resource "openstack_networking_router_interface_v2" "router_intf_chi" {
  provider  = openstack.chi
  router_id = openstack_networking_router_v2.router_chi.id
  subnet_id = openstack_networking_subnet_v2.private_subnet_chi.id
}

#############################################
# 9) GPU VM on CHI@TACC
#############################################
# resource "openstack_compute_instance_v2" "gpu_node" {
#   provider    = openstack.chi
#   name        = "gpu-node-project4-${var.user_tag}"
#   flavor_name = var.gpu_flavor
#   key_pair    = openstack_compute_keypair_v2.keypair.name

#   # ────────────────────────────────────────────────────────────────
#   # Required for Ironic: config drive + local disk mapping
#   # ────────────────────────────────────────────────────────────────
#   config_drive = true

#   block_device {
#     uuid                  = var.gpu_image_id # Glance image UUID
#     source_type           = "image"          # pull from Glance
#     destination_type      = "local"          # write to bare‐metal’s local disk
#     boot_index            = 0                # primary boot device
#     delete_on_termination = true             # cleanup when server is destroyed
#   }

#   network {
#     uuid = openstack_networking_network_v2.private_net_chi.id
#   }

#   scheduler_hints {
#     additional_properties = {
#       reservation = var.gpu_reservation_id
#     }
#   }
# }
data "openstack_compute_instance_v2" "gpu_node" {
  provider = openstack.chi
  name     = "gpu-node-project4-${var.user_tag}"
}


#############################################
# 10) Optional staging VM
#############################################
resource "openstack_compute_instance_v2" "staging_node" {
  count       = var.enable_staging ? 1 : 0
  provider    = openstack.chi
  name        = "staging-node-${var.user_tag}"
  image_name  = var.staging_image
  flavor_name = var.staging_flavor
  key_pair    = openstack_compute_keypair_v2.keypair.name

  security_groups = [
    openstack_networking_secgroup_v2.mlops_secgrp.id
  ]

  network {
    # uuid = openstack_networking_network_v2.private_subnet_chi.id
    uuid = openstack_networking_network_v2.private_net_chi.id
  }
}

#############################################
# 11) Floating IPs & associations
#############################################
resource "openstack_networking_floatingip_v2" "fip_services" {
  provider = openstack.kvm
  pool     = data.openstack_networking_network_v2.external_kvm.name
}

resource "openstack_compute_floatingip_associate_v2" "assoc_services" {
  provider    = openstack.kvm
  floating_ip = openstack_networking_floatingip_v2.fip_services.address
  instance_id = openstack_compute_instance_v2.services_node.id
}

resource "openstack_networking_floatingip_v2" "fip_gpu" {
  provider = openstack.chi
  pool     = data.openstack_networking_network_v2.external_chi.name
}

resource "openstack_compute_floatingip_associate_v2" "assoc_gpu" {
  provider    = openstack.chi
  floating_ip = openstack_networking_floatingip_v2.fip_gpu.address
  instance_id = openstack_compute_instance_v2.gpu_node.id
}

resource "openstack_networking_floatingip_v2" "fip_staging" {
  count    = var.enable_staging ? 1 : 0
  provider = openstack.chi
  pool     = data.openstack_networking_network_v2.external_chi.name
}

resource "openstack_compute_floatingip_associate_v2" "assoc_staging" {
  count       = var.enable_staging ? 1 : 0
  provider    = openstack.chi
  floating_ip = openstack_networking_floatingip_v2.fip_staging[count.index].address
  instance_id = openstack_compute_instance_v2.staging_node[count.index].id
}
#############################################