# devops/terraform/outputs.tf

output "vm_name" {
  description = "Name of the created VM instance"
  value       = openstack_compute_instance_v2.mlops_vm.name
}

output "vm_id" {
  description = "ID of the created VM instance"
  value       = openstack_compute_instance_v2.mlops_vm.id
}

output "vm_private_ip" {
  description = "Private IP address of the VM instance"
  value       = openstack_compute_instance_v2.mlops_vm.access_ip_v4 # Or use networks[0].fixed_ip_v4
}

output "vm_floating_ip" {
  description = "Floating (public) IP address associated with the VM"
  value       = openstack_networking_floatingip_v2.mlops_fip.address
}

output "ssh_command" {
  description = "SSH command to connect to the VM using the floating IP"
  value       = "ssh -i ~/.ssh/${var.chameleon_keypair_name}.pem cc@${openstack_networking_floatingip_v2.mlops_fip.address}"
}