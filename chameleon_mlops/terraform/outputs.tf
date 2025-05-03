# output "services_vm_ip" {
#   description = "Floating IP of the services VM"
#   value       = openstack_networking_floatingip_v2.fip_services.address
# }
# output "gpu_vm_ip" {
#   description = "Floating IP of the GPU VM"
#   value       = openstack_networking_floatingip_v2.fip_gpu.address
# }
# output "internal_ips" {
#   description = "Internal IPs of the instances"
#   value = {
#     services_node = "192.168.100.50"
#     gpu_node      = "192.168.100.51"
#   }
# }

output "services_node_ip" {
  value = openstack_networking_floatingip_v2.fip_services.address
}
output "gpu_node_ip" {
  value = openstack_networking_floatingip_v2.fip_gpu.address
}
output "services_private_ip" {
  value = openstack_compute_instance_v2.services_node.network[0].fixed_ip_v4
}
output "gpu_private_ip" {
  value = openstack_compute_instance_v2.gpu_node.network[0].fixed_ip_v4
}

# Optionally, output an Ansible inventory snippet
output "ansible_inventory" {
  description = "Suggested inventory snippet for Ansible"
  value = <<EOL
[services]
services-node ansible_host=${openstack_networking_floatingip_v2.fip_services.address} ansible_user=ubuntu

[gpu]
gpu-node ansible_host=${openstack_networking_floatingip_v2.fip_gpu.address} ansible_user=ubuntu

[all:vars]
ansible_ssh_private_key_file=~/.ssh/mlops_proj_key
ansible_python_interpreter=/usr/bin/python3
EOL
}
