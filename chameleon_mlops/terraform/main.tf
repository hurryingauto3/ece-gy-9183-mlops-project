// Data source for external (public) network (to get its ID)
data "openstack_networking_network_v2" "external" {
  name = "public"
}

// Networking: private network, subnet, and router for the project
resource "openstack_networking_network_v2" "private_net" {
  name           = "mlops-net"
  admin_state_up = true
}
resource "openstack_networking_subnet_v2" "private_subnet" {
  name           = "mlops-subnet"
  network_id     = openstack_networking_network_v2.private_net.id
  cidr           = "192.168.100.0/24"
  ip_version     = 4
  gateway_ip     = "192.168.100.1"
  allocation_pools {
    start = "192.168.100.50"
    end   = "192.168.100.200"
  }
  dns_nameservers = ["8.8.8.8", "8.8.4.4"]
}
resource "openstack_networking_router_v2" "router" {
  name                = "mlops-router"
  admin_state_up      = true
  external_network_id = data.openstack_networking_network_v2.external.id
}
resource "openstack_networking_router_interface_v2" "router_interface" {
  router_id = openstack_networking_router_v2.router.id
  subnet_id = openstack_networking_subnet_v2.private_subnet.id
}

// Security Group: allow SSH, web UIs, and internal communication
resource "openstack_networking_secgroup_v2" "mlops_secgrp" {
  name        = "mlops-secgroup"
  description = "Security group for MLOps VMs"
}
resource "openstack_networking_secgroup_rule_v2" "ssh_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
}
resource "openstack_networking_secgroup_rule_v2" "icmp_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "icmp"
  remote_ip_prefix  = "0.0.0.0/0"
}
resource "openstack_networking_secgroup_rule_v2" "services_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_group_id   = openstack_networking_secgroup_v2.mlops_secgrp.id
}
# (The above rule allows all traffic between instances in this secgroup internally. 
# Next, specific external-facing ports for web services:)
resource "openstack_networking_secgroup_rule_v2" "mlflow_ui_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 5000
  port_range_max    = 5000
  remote_ip_prefix  = "0.0.0.0/0"
}
resource "openstack_networking_secgroup_rule_v2" "grafana_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 3000
  port_range_max    = 3000
  remote_ip_prefix  = "0.0.0.0/0"
}
resource "openstack_networking_secgroup_rule_v2" "prometheus_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 9090
  port_range_max    = 9090
  remote_ip_prefix  = "0.0.0.0/0"
}
resource "openstack_networking_secgroup_rule_v2" "minio_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 9000
  port_range_max    = 9001
  remote_ip_prefix  = "0.0.0.0/0"
}
resource "openstack_networking_secgroup_rule_v2" "fastapi_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 8000
  port_range_max    = 8000
  remote_ip_prefix  = "0.0.0.0/0"
}
resource "openstack_networking_secgroup_rule_v2" "ray_dashboard_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 8265
  port_range_max    = 8265
  remote_ip_prefix  = "0.0.0.0/0"
}
# (Port 10001 for Ray client server, if needed to connect via ray client API)
resource "openstack_networking_secgroup_rule_v2" "ray_client_ingress" {
  security_group_id = openstack_networking_secgroup_v2.mlops_secgrp.id
  direction         = "ingress"
  protocol          = "tcp"
  port_range_min    = 10001
  port_range_max    = 10001
  remote_ip_prefix  = "0.0.0.0/0"
}

// SSH Key Pair: import the shared project public key to OpenStack
resource "openstack_compute_keypair_v2" "project_key" {
  name       = var.ssh_key_name
  public_key = file(var.public_key_path)
}

// Server: Services VM (for MLflow, DB, Prometheus, Grafana, etc.)
resource "openstack_compute_instance_v2" "services_node" {
  name            = "mlops-services-node"
  flavor_name     = var.services_flavor
  image_name      = var.services_image
  key_pair        = openstack_compute_keypair_v2.project_key.name
  security_groups = [openstack_networking_secgroup_v2.mlops_secgrp.name]
  network {
    uuid           = openstack_networking_network_v2.private_net.id
    fixed_ip_v4    = "192.168.100.50"  # assign a fixed IP for internal reference (optional)
  }
}

// Server: GPU node (bare metal)
resource "openstack_compute_instance_v2" "gpu_node" {
  name        = "mlops-gpu-node"
  flavor_name = var.gpu_flavor            # "baremetal" flavor for reserved bare metal
  image_name  = var.gpu_image
  key_pair    = openstack_compute_keypair_v2.project_key.name
  # Note: security groups are not applied on baremetal instances (no neutron filtering)&#8203;:contentReference[oaicite:11]{index=11}.
  # We still attach it for consistency (it will be ignored for filtering but used for our internal rules logic).
  security_groups = [openstack_networking_secgroup_v2.mlops_secgrp.name]
  network {
    uuid           = openstack_networking_network_v2.private_net.id
    fixed_ip_v4    = "192.168.100.51"  # fixed internal IP for GPU node
  }
  scheduler_hints = { 
    reservation = var.gpu_reservation_id  # use the reserved A100 node from lease&#8203;:contentReference[oaicite:12]{index=12}
  }
}

// Allocate Floating IPs for external access
resource "openstack_networking_floatingip_v2" "fip_services" {
  pool = data.openstack_networking_network_v2.external.name
}
resource "openstack_networking_floatingip_v2" "fip_gpu" {
  pool = data.openstack_networking_network_v2.external.name
}

// Associate Floating IPs with the instances
resource "openstack_compute_floatingip_associate_v2" "fip_attach_services" {
  floating_ip = openstack_networking_floatingip_v2.fip_services.address
  instance_id = openstack_compute_instance_v2.services_node.id
}
resource "openstack_compute_floatingip_associate_v2" "fip_attach_gpu" {
  floating_ip = openstack_networking_floatingip_v2.fip_gpu.address
  instance_id = openstack_compute_instance_v2.gpu_node.id
}
