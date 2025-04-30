### USERNAME PWD...
provider "openstack" {
  auth_url    = "https://kvm.tacc.chameleoncloud.org:5000/v3"
  tenant_name = "CH-251409"  
  user_name   = "__"
  password    = "__"
  region      = "RegionOne"
  domain_name = "default"
}

data "openstack_networking_network_v2" "external" {
  name = "public"
}

resource "openstack_compute_keypair_v2" "project_key" {
  name       = "project4_key"
  public_key = file("~/.ssh/project4_key.pub")
}

resource "openstack_networking_network_v2" "project_net" {
  name = "net-project4"
}

resource "openstack_networking_subnet_v2" "project_subnet" {
  name       = "subnet-project4"
  network_id = openstack_networking_network_v2.project_net.id
  cidr       = "192.168.100.0/24"
  ip_version = 4
}

resource "openstack_networking_router_v2" "project_router" {
  name                = "router-project4"
  external_network_id = data.openstack_networking_network_v2.external.id
}

resource "openstack_networking_router_interface_v2" "project_router_interface" {
  router_id = openstack_networking_router_v2.project_router.id
  subnet_id = openstack_networking_subnet_v2.project_subnet.id
}

resource "openstack_networking_secgroup_v2" "project_secgroup" {
  name        = "secgroup-project4"
  description = "Security group for SSH and ICMP"
}

resource "openstack_networking_secgroup_rule_v2" "ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.project_secgroup.id
}

resource "openstack_networking_secgroup_rule_v2" "icmp" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "icmp"
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.project_secgroup.id
}

resource "openstack_blockstorage_volume_v3" "project_volume" {
  name = "volume-project4"
  size = 10
}

resource "openstack_compute_instance_v2" "vm" {
  name            = "vm1-project4"
  image_name      = "CC-Ubuntu20.04"
  flavor_name     = "gpu.medium"  # Change if you want more GPU/CPU
  key_pair        = openstack_compute_keypair_v2.project_key.name
  security_groups = [openstack_networking_secgroup_v2.project_secgroup.name]

  network {
    uuid = openstack_networking_network_v2.project_net.id
  }

  user_data = file("${path.module}/cloud-init-mlflow.yaml")
}

resource "openstack_compute_volume_attach_v2" "attach_volume" {
  instance_id = openstack_compute_instance_v2.vm.id
  volume_id   = openstack_blockstorage_volume_v3.project_volume.id
}

resource "openstack_networking_floatingip_v2" "fip" {
  pool = data.openstack_networking_network_v2.external.name
}

resource "openstack_compute_floatingip_associate_v2" "fip_assoc" {
  floating_ip = openstack_networking_floatingip_v2.fip.address
  instance_id = openstack_compute_instance_v2.vm.id
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/project4_key ubuntu@${openstack_networking_floatingip_v2.fip.address}"
}