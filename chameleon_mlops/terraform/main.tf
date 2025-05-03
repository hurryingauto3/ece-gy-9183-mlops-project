#############################################
#                main.tf
#############################################

# ──────────────────────────────────────────
# 1) External (public) network data sources
# ──────────────────────────────────────────
data "openstack_networking_network_v2" "external_kvm" {
  provider = openstack.kvm
  name     = var.ext_net_name_kvm # e.g. "sharednet4"
}

data "openstack_networking_network_v2" "external_chi" {
  provider = openstack.chi
  name     = var.ext_net_name_chi # e.g. "public"
}

# ──────────────────────────────────────────
# 2) SSH Keypair import (KVM@TACC)
# ──────────────────────────────────────────
resource "openstack_compute_keypair_v2" "keypair" {
  provider   = openstack.kvm
  name       = var.keypair_name
  public_key = file(var.public_key_path)
}

# ──────────────────────────────────────────
# 3) Security Group + rules (KVM@TACC)
# ──────────────────────────────────────────
resource "openstack_networking_secgroup_v2" "mlops_secgrp" {
  provider    = openstack.kvm
  name        = var.security_group_name
  description = "Security group for MLOps VMs"
}

# SSH
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

# ICMP (ping)
resource "openstack_networking_secgroup_rule_v2" "icmp_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "icmp"
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# Intra‑group traffic (all TCP)
resource "openstack_networking_secgroup_rule_v2" "internal" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_group_id   = openstack_networking_secgroup_v2.mlops_secgrp.id
  ethertype         = "IPv4"
}

# MLflow UI
resource "openstack_networking_secgroup_rule_v2" "mlflow_ui_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 5000
  port_range_max    = 5000
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# Grafana
resource "openstack_networking_secgroup_rule_v2" "grafana_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 3000
  port_range_max    = 3000
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# Prometheus
resource "openstack_networking_secgroup_rule_v2" "prometheus_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 9090
  port_range_max    = 9090
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# MinIO (9000 & 9001)
resource "openstack_networking_secgroup_rule_v2" "minio_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 9000
  port_range_max    = 9001
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# FastAPI
resource "openstack_networking_secgroup_rule_v2" "fastapi_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 8000
  port_range_max    = 8000
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# Ray Dashboard
resource "openstack_networking_secgroup_rule_v2" "ray_dashboard_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 8265
  port_range_max    = 8265
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# Ray Client Server
resource "openstack_networking_secgroup_rule_v2" "ray_client_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 10001
  port_range_max    = 10001
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# ──────────────────────────────────────────
# 4) Networking on KVM@TACC (private subnet + router)
# ──────────────────────────────────────────
resource "openstack_networking_network_v2" "private_net_kvm" {
  provider       = openstack.kvm
  name           = var.network_name
  admin_state_up = true
}

resource "openstack_networking_subnet_v2" "private_subnet_kvm" {
  provider   = openstack.kvm
  name       = "${var.network_name}-subnet"
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
  name                = "${var.network_name}-router"
  admin_state_up      = true
  external_network_id = data.openstack_networking_network_v2.external_kvm.id
}

resource "openstack_networking_router_interface_v2" "router_intf_kvm" {
  provider  = openstack.kvm
  router_id = openstack_networking_router_v2.router_kvm.id
  subnet_id = openstack_networking_subnet_v2.private_subnet_kvm.id
}

# ──────────────────────────────────────────
# 5) Services VM on KVM@TACC
# ──────────────────────────────────────────
resource "openstack_compute_instance_v2" "services_node" {
  provider        = openstack.kvm
  name            = "services-node"
  image_name      = var.services_image
  flavor_name     = var.services_flavor
  key_pair        = openstack_compute_keypair_v2.keypair.name
  security_groups = [openstack_networking_secgroup_v2.mlops_secgrp.name]

  network {
    uuid = openstack_networking_network_v2.private_net_kvm.id
  }
}

# ──────────────────────────────────────────
# 6) (Optional) Private net on CHI@TACC for GPU & staging
# ──────────────────────────────────────────
resource "openstack_networking_network_v2" "private_net_chi" {
  provider              = openstack.chi
  name                  = "${var.network_name}-chi"
  admin_state_up        = true
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet_chi" {
  provider   = openstack.chi
  name       = "${var.network_name}-chi-subnet"
  network_id = openstack_networking_network_v2.private_net_chi.id
  cidr       = var.network_cidr # or introduce var.network_cidr_chi
  ip_version = 4
  gateway_ip = var.network_gateway # adjust if needed
  allocation_pool {
    start = var.network_pool_start
    end   = var.network_pool_end
  }
  dns_nameservers = var.dns_nameservers
}

# ──────────────────────────────────────────
# 7) Security Group + rules (CHI@TACC)
# ──────────────────────────────────────────
# CHI@TACC security group (new)
# This leads to an error: Quota exceeded for resources: ['security_group']
# resource "openstack_networking_secgroup_v2" "mlops_secgrp_chi" {
#   provider    = openstack.chi
#   name        = var.security_group_name
#   description = "Security group for MLOps GPU VM"
# }

# data "openstack_networking_secgroup_v2" "mlops_secgrp_chi" {
#   provider = openstack.chi
#   name     = var.security_group_name
# }

# SSH
# resource "openstack_networking_secgroup_rule_v2" "ssh_ingress_chi" {
#   provider = openstack.chi
#   # security_group_id = openstack_networking_secgroup_v2.mlops_secgrp_chi.id
#   # security_group_id = data.openstack_networking_secgroup_v2.mlops_secgrp_chi.id
#   direction        = "ingress"
#   ethertype        = "IPv4"
#   protocol         = "tcp"
#   port_range_min   = 22
#   port_range_max   = 22
#   remote_ip_prefix = "0.0.0.0/0"
# }

# ──────────────────────────────────────────
# 7) Private subnet (CHI@TACC))
# ──────────────────────────────────────────
# 1) Router in CHI@TACC
resource "openstack_networking_router_v2" "router_chi" {
  provider            = openstack.chi
  name                = "${var.network_name}-router-chi"
  admin_state_up      = true
  external_network_id = data.openstack_networking_network_v2.external_chi.id
}

# 2) Hook CHI subnet into that router
resource "openstack_networking_router_interface_v2" "router_intf_chi" {
  provider  = openstack.chi
  router_id = openstack_networking_router_v2.router_chi.id
  subnet_id = openstack_networking_subnet_v2.private_subnet_chi.id
}

# ──────────────────────────────────────────
# 8) GPU VM on CHI@TACC
# ──────────────────────────────────────────
resource "openstack_compute_instance_v2" "gpu_node" {
  provider    = openstack.chi
  name        = "gpu-node"
  image_name  = var.gpu_image
  flavor_name = var.gpu_flavor
  key_pair    = openstack_compute_keypair_v2.keypair.name
  # security_groups = [data.openstack_networking_secgroup_v2.mlops_secgrp_chi.name]
  # no security_groups declared → port_security_disabled on this network

  network {
    uuid = openstack_networking_network_v2.private_net_chi.id
  }

  scheduler_hints {
    additional_properties = {
      reservation = var.gpu_reservation_id
    }
  }
}

# ──────────────────────────────────────────
# 9) Optional staging VM on CHI@TACC
# ──────────────────────────────────────────
resource "openstack_compute_instance_v2" "staging_node" {
  count           = var.enable_staging ? 1 : 0
  provider        = openstack.chi
  name            = "staging-node"
  image_name      = var.staging_image
  flavor_name     = var.staging_flavor
  key_pair        = openstack_compute_keypair_v2.keypair.name
  security_groups = [openstack_networking_secgroup_v2.mlops_secgrp.name]

  network {
    uuid = openstack_networking_network_v2.private_net_chi.id
  }
}

# ──────────────────────────────────────────
# 10) Floating IPs & associations
# ──────────────────────────────────────────
# Services node
resource "openstack_networking_floatingip_v2" "fip_services" {
  provider = openstack.kvm
  pool     = data.openstack_networking_network_v2.external_kvm.name
}
resource "openstack_compute_floatingip_associate_v2" "assoc_services" {
  provider    = openstack.kvm
  floating_ip = openstack_networking_floatingip_v2.fip_services.address
  instance_id = openstack_compute_instance_v2.services_node.id
}

# GPU node
resource "openstack_networking_floatingip_v2" "fip_gpu" {
  provider = openstack.chi
  pool     = data.openstack_networking_network_v2.external_chi.name
}
resource "openstack_compute_floatingip_associate_v2" "assoc_gpu" {
  provider    = openstack.chi
  floating_ip = openstack_networking_floatingip_v2.fip_gpu.address
  instance_id = openstack_compute_instance_v2.gpu_node.id
}

# Staging node (if enabled)
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
