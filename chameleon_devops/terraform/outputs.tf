
output "services_node_ip" {
  value = openstack_networking_floatingip_v2.fip_services.address
}

output "ssh_command_services_node" {
  value = "ssh -i ~/.ssh/${openstack_compute_keypair_v2.keypair_kvm.name} cc@${openstack_networking_floatingip_v2.fip_services.address}"
}

output "gpu_node_ip" {
  value = local.use_chi_tacc ? openstack_networking_floatingip_v2.fip_gpu_chi_tacc[0].address : openstack_networking_floatingip_v2.fip_gpu_chi_uc[0].address
}

output "ssh_command_gpu_node" {
  value = local.use_chi_tacc ? "ssh -i ${local.private_key_path} cc@${openstack_networking_floatingip_v2.fip_gpu_chi_tacc[0].address}" : "ssh -i ${local.private_key_path} cc@${openstack_networking_floatingip_v2.fip_gpu_chi_uc[0].address}"
}

output "services_private_ip" {
  value = openstack_compute_instance_v2.services_node.network[0].fixed_ip_v4
}

output "gpu_private_ip" {
  value = local.use_chi_tacc ? openstack_compute_instance_v2.gpu_node_chi_tacc[0].network[0].fixed_ip_v4 : openstack_compute_instance_v2.gpu_node_chi_uc[0].network[0].fixed_ip_v4
}

output "ansible_inventory" {
  value = <<-EOT
    [services_nodes]
    services-node ansible_host=${openstack_networking_floatingip_v2.fip_services.address} ansible_user=cc

    [gpu_nodes]
    gpu-node ansible_host=${local.use_chi_tacc ? openstack_networking_floatingip_v2.fip_gpu_chi_tacc[0].address : openstack_networking_floatingip_v2.fip_gpu_chi_uc[0].address} ansible_user=cc

    [k8s_control_plane]
    services-node ansible_host=${openstack_networking_floatingip_v2.fip_services.address} ansible_user=cc

    [all:vars]
    ansible_ssh_private_key_file=${local.private_key_path}
    ansible_python_interpreter=/usr/bin/python3
  EOT
}
