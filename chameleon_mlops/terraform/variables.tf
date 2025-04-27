variable "openstack_cloud" {
  description = "Name of the OpenStack cloud (as defined in clouds.yaml)"
  type        = string
  default     = "chameleon-uc"  # e.g., entry for CHI@UC in clouds.yaml
}
variable "ssh_key_name" {
  description = "Name to assign to the imported SSH key in OpenStack"
  type        = string
  default     = "mlops-project-key"
}
variable "public_key_path" {
  description = "Path to the public SSH key file to import"
  type        = string
  default     = "~/.ssh/mlops_key.pub"  # update if your key path is different
}
variable "gpu_reservation_id" {
  description = "Reservation ID of the leased GPU node (from Chameleon lease)"
  type        = string
  default     = ""  # e.g., "da123456-...-abcdef" (leave blank if not using a lease)
}
variable "services_flavor" {
  description = "OpenStack flavor for the general services VM"
  type        = string
  default     = "m1.large"  # e.g., 4 VCPUs, 8GB RAM (adjust based on needs/quotas)
}
variable "gpu_flavor" {
  description = "OpenStack flavor for the GPU node (baremetal)"
  type        = string
  default     = "baremetal"  # Chameleon uses "baremetal" flavor for any reserved baremetal node&#8203;:contentReference[oaicite:9]{index=9}
}
variable "services_image" {
  description = "Image name for services VM"
  type        = string
  default     = "CC-Ubuntu20.04"  # Chameleon Ubuntu 20.04 image
}
variable "gpu_image" {
  description = "Image name for GPU VM (with CUDA/driver support)"
  type        = string
  default     = "CC-Ubuntu20.04-CUDA11"  # image with Nvidia drivers pre-installed&#8203;:contentReference[oaicite:10]{index=10}
}
