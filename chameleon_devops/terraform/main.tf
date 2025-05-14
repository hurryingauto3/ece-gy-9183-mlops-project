#############################################
#                main.tf                     #
#############################################

locals {
  use_chi_tacc = var.gpu_region == var.chi_region
  use_chi_uc   = var.gpu_region == var.uc_region

  // must be strings for for_each
  app_ports_str = [
    "5000", "3000", "9090",
    "9000", "9001", "8000",
    "8265", "10001", "8501",
    "80", "443", "8089",
  ]

  # path to your private SSH key (drops the ".pub")
  private_key_path = "~/.ssh/${var.keypair_name}"
}

# ──────────────────────────────────────────
# 1) External (public) network data sources
# ──────────────────────────────────────────
data "openstack_networking_network_v2" "external_kvm" {
  provider = openstack.kvm
  name     = var.ext_net_name_kvm
}

data "openstack_networking_network_v2" "external_uc" {
  count    = local.use_chi_uc ? 1 : 0
  provider = openstack.uc
  name     = var.ext_net_name_uc
}

data "openstack_networking_network_v2" "external_chi" {
  count    = local.use_chi_tacc ? 1 : 0
  provider = openstack.chi
  name     = var.ext_net_name_tacc
}

# ──────────────────────────────────────────
# 2) SSH Keypair import
# ──────────────────────────────────────────
resource "openstack_compute_keypair_v2" "keypair_kvm" {
  provider   = openstack.kvm
  name       = var.keypair_name
  public_key = file(var.public_key_path)
}

resource "openstack_compute_keypair_v2" "keypair_uc" {
  provider   = openstack.uc
  name       = var.keypair_name
  public_key = file(var.public_key_path)
}

resource "openstack_compute_keypair_v2" "keypair_chi" {
  provider   = openstack.chi
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
resource "openstack_networking_secgroup_v2" "mlops_secgrp_proj4" {
  provider    = openstack.kvm
  name        = "mlops-secgrp-proj4" # must be globally unique in your project
  description = "Security group for MLOps VMs (proj4)"
}

# Ingress rules
resource "openstack_networking_secgroup_rule_v2" "ssh_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp_proj4.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

resource "openstack_networking_secgroup_rule_v2" "icmp_ingress" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp_proj4.id
  direction         = "ingress"
  protocol          = "icmp"
  remote_ip_prefix  = "0.0.0.0/0"
  ethertype         = "IPv4"
}

# Intra-group traffic
resource "openstack_networking_secgroup_rule_v2" "internal" {
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp_proj4.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_group_id   = openstack_networking_secgroup_v2.mlops_secgrp_proj4.id
  ethertype         = "IPv4"
}

// App‑specific ports (loop)
resource "openstack_networking_secgroup_rule_v2" "app_ports" {
  for_each          = toset(local.app_ports_str)
  provider          = openstack.kvm
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp_proj4.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = tonumber(each.value)
  port_range_max    = tonumber(each.value)
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
  name            = "services-node-${var.network_name}"
  image_name      = var.services_image
  flavor_name     = var.services_flavor
  key_pair        = openstack_compute_keypair_v2.keypair_kvm.name
  security_groups = [openstack_networking_secgroup_v2.mlops_secgrp_proj4.name]

  network {
    uuid = openstack_networking_network_v2.private_net_kvm.id
  }
}

# ──────────────────────────────────────────
# 6) Private net on CHI@TACC for GPU & staging
#    (use existing network to avoid stale-ID errors)
# ──────────────────────────────────────────
resource "openstack_networking_network_v2" "private_net_chi" {
  count = local.use_chi_tacc ? 1 : 0
  provider              = openstack.chi
  name                  = "${var.network_name}-chi"
  admin_state_up        = true
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet_chi" {
  count = local.use_chi_tacc ? 1 : 0
  provider   = openstack.chi
  name       = "${var.network_name}-chi-subnet"
  network_id = openstack_networking_network_v2.private_net_chi[count.index].id
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
  count = local.use_chi_tacc ? 1 : 0
  provider            = openstack.chi
  name                = "${var.network_name}-router-chi"
  admin_state_up      = true
  external_network_id = data.openstack_networking_network_v2.external_chi[0].id
}

resource "openstack_networking_router_interface_v2" "router_intf_chi" {
  count = local.use_chi_tacc ? 1 : 0
  provider  = openstack.chi
  router_id = openstack_networking_router_v2.router_chi[count.index].id
  # subnet_id = data.openstack_networking_subnet_v2.private_subnet_chi.id
  subnet_id = openstack_networking_subnet_v2.private_subnet_chi[count.index].id
}

# ──────────────────────────────────────────
# 7) Private net on CHI@UC (GPU fallback)
# ──────────────────────────────────────────
resource "openstack_networking_network_v2" "private_net_uc" {
  count = local.use_chi_uc ? 1 : 0
  provider              = openstack.uc
  name                  = "${var.network_name}-uc"
  admin_state_up        = true
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet_uc" {
  count = local.use_chi_uc ? 1 : 0
  provider   = openstack.uc
  name       = "${var.network_name}-uc-subnet"
  network_id = openstack_networking_network_v2.private_net_uc[count.index].id
  cidr       = var.network_cidr
  ip_version = 4
  gateway_ip = var.network_gateway
  allocation_pool {
    start = var.network_pool_start
    end   = var.network_pool_end
  }
  dns_nameservers = var.dns_nameservers
}

resource "openstack_networking_router_v2" "router_uc" {
  count = local.use_chi_uc ? 1 : 0
  provider            = openstack.uc
  name                = "${var.network_name}-router-uc"
  admin_state_up      = true
  external_network_id = data.openstack_networking_network_v2.external_uc[0].id
}

resource "openstack_networking_router_interface_v2" "router_intf_uc" {
  count = local.use_chi_uc ? 1 : 0
  provider  = openstack.uc
  router_id = openstack_networking_router_v2.router_uc[count.index].id
  subnet_id = openstack_networking_subnet_v2.private_subnet_uc[count.index].id
}

# ──────────────────────────────────────────
# 8) GPU VM(s) on CHI depending on region
# ──────────────────────────────────────────
resource "openstack_compute_instance_v2" "gpu_node_chi_tacc" {
  count        = local.use_chi_tacc ? 1 : 0
  provider     = openstack.chi
  name         = "gpu-node-${var.network_name}"
  image_id     = var.gpu_image_id_tacc
  flavor_name  = var.gpu_flavor
  key_pair     = openstack_compute_keypair_v2.keypair_chi.name
  config_drive = true

  block_device {
    uuid                  = var.gpu_image_id_tacc
    source_type           = "image"
    destination_type      = "local"
    boot_index            = 0
    delete_on_termination = true
  }

  network {
    # uuid = data.openstack_networking_network_v2.private_net_chi.id
    uuid = openstack_networking_network_v2.private_net_chi[count.index].id
  }

  scheduler_hints {
    additional_properties = {
      reservation = var.gpu_reservation_id_tacc
    }
  }

  depends_on = [openstack_networking_subnet_v2.private_subnet_chi]
  # depends_on = [data.openstack_networking_subnet_v2.private_subnet_chi]
}

resource "openstack_compute_instance_v2" "gpu_node_chi_uc" {
  count        = local.use_chi_uc ? 1 : 0
  provider     = openstack.uc
  name         = "gpu-node-${var.network_name}"
  image_id     = var.gpu_image_id_uc
  flavor_name  = var.gpu_flavor
  key_pair     = openstack_compute_keypair_v2.keypair_uc.name
  config_drive = true

  block_device {
    uuid                  = var.gpu_image_id_uc
    source_type           = "image"
    destination_type      = "local"
    boot_index            = 0
    delete_on_termination = true
  }

  network {
    uuid = openstack_networking_network_v2.private_net_uc[count.index].id
  }

  scheduler_hints {
    additional_properties = {
      reservation = var.gpu_reservation_id_uc
    }
  }
}

# ──────────────────────────────────────────
# 9) Optional staging VM on CHI@TACC
# ──────────────────────────────────────────
# resource "openstack_compute_instance_v2" "staging_node" {
#   count       = var.enable_staging ? 1 : 0
#   provider    = openstack.chi
#   name        = "staging-node-${var.network_name}"
#   image_name  = var.staging_image
#   flavor_name = var.staging_flavor
#   key_pair    = openstack_compute_keypair_v2.keypair_chi.name

#   network {
#     uuid = data.openstack_networking_network_v2.private_net_chi.id
#   }
# }

# ──────────────────────────────────────────
# 10) Floating IPs & associations
# ──────────────────────────────────────────
resource "openstack_networking_floatingip_v2" "fip_services" {
  provider = openstack.kvm
  pool     = data.openstack_networking_network_v2.external_kvm.name
}

resource "openstack_compute_floatingip_associate_v2" "assoc_services" {
  provider    = openstack.kvm
  floating_ip = openstack_networking_floatingip_v2.fip_services.address
  instance_id = openstack_compute_instance_v2.services_node.id
}

resource "openstack_networking_floatingip_v2" "fip_gpu_chi_tacc" {
  count    = local.use_chi_tacc ? 1 : 0
  provider = openstack.chi
  pool     = data.openstack_networking_network_v2.external_chi[count.index].name
}

resource "openstack_compute_floatingip_associate_v2" "assoc_gpu_chi_tacc" {
  count       = local.use_chi_tacc ? 1 : 0
  provider    = openstack.chi
  floating_ip = openstack_networking_floatingip_v2.fip_gpu_chi_tacc[0].address
  instance_id = openstack_compute_instance_v2.gpu_node_chi_tacc[0].id
}

resource "openstack_networking_floatingip_v2" "fip_gpu_chi_uc" {
  count    = local.use_chi_uc ? 1 : 0
  provider = openstack.uc
  pool     = data.openstack_networking_network_v2.external_uc[count.index].name
}

resource "openstack_compute_floatingip_associate_v2" "assoc_gpu_chi_uc" {
  count       = local.use_chi_uc ? 1 : 0
  provider    = openstack.uc
  floating_ip = openstack_networking_floatingip_v2.fip_gpu_chi_uc[0].address
  instance_id = openstack_compute_instance_v2.gpu_node_chi_uc[0].id
}

# resource "openstack_networking_floatingip_v2" "fip_staging" {
#   count    = var.enable_staging ? 1 : 0
#   provider = openstack.chi
#   pool     = data.openstack_networking_network_v2.external_chi.name
# }

# resource "openstack_compute_floatingip_associate_v2" "assoc_staging" {
#   count       = var.enable_staging ? 1 : 0
#   provider    = openstack.chi
#   floating_ip = openstack_networking_floatingip_v2.fip_staging[count.index].address
#   instance_id = openstack_compute_instance_v2.staging_node[count.index].id
# }

# ──────────────────────────────────────────
# 11) Persistent storage
# ──────────────────────────────────────────
resource "openstack_blockstorage_volume_v3" "persistent_volume_services" {
  provider = openstack.kvm
  size     = 100
  name     = "persistent-storage-services"
}

resource "openstack_compute_volume_attach_v2" "attach_persistent_services" {
  provider    = openstack.kvm
  instance_id = openstack_compute_instance_v2.services_node.id
  volume_id   = openstack_blockstorage_volume_v3.persistent_volume_services.id
  device      = "/dev/vdb"
}

resource "openstack_blockstorage_volume_v3" "persistent_volume_gpu_uc" {
  count    = local.use_chi_uc && var.enable_gpu_block_storage ? 1 : 0
  provider = openstack.uc
  size     = 100
  name     = "gpu-persistent-volume"
}

resource "openstack_compute_volume_attach_v2" "attach_gpu_volume_uc" {
  count       = local.use_chi_uc && var.enable_gpu_block_storage ? 1 : 0
  provider    = openstack.uc
  instance_id = openstack_compute_instance_v2.gpu_node_chi_uc[0].id
  volume_id   = openstack_blockstorage_volume_v3.persistent_volume_gpu_uc[0].id
}

resource "openstack_blockstorage_volume_v3" "persistent_volume_gpu_tacc" {
  count    = local.use_chi_tacc && var.enable_gpu_block_storage ? 1 : 0
  provider = openstack.chi
  size     = 100
  name     = "gpu-persistent-volume"
}

resource "openstack_compute_volume_attach_v2" "attach_gpu_volume_tacc" {
  count       = local.use_chi_tacc && var.enable_gpu_block_storage ? 1 : 0
  provider    = openstack.chi
  instance_id = openstack_compute_instance_v2.gpu_node_chi_tacc[0].id
  volume_id   = openstack_blockstorage_volume_v3.persistent_volume_gpu_tacc[0].id
}
