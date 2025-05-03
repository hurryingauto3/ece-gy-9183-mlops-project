# OpenStack / Chameleon settings
type = string
variable "cloud" {
  description = "OpenStack cloud name (from clouds.yaml or RC file)"
  type        = string
  default     = "chameleon"
}
variable "services_region" {
  description = "Region for the Services node (KVM@TACC)"
  type        = string
  default     = "CHI@TACC"
}
variable "gpu_region" {
  description = "Region for the GPU node (CHI@UC or CHI@TACC)"
  type        = string
  default     = "CHI@UC"
}

# External network (floating IP pool)
variable "ext_net_name" {
  description = "Name of the external/public network for floating IPs"
  type        = string
  default     = "public-ext-net"
}

# VM Images & Flavors
variable "services_image" {
  description = "Image name for Services node (e.g., CC-Ubuntu20.04)"
  type        = string
  default     = "CC-Ubuntu24.04-CUDA"  # image with Nvidia drivers pre-installed&#8203;:contentReference[oaicite:10]{index=10}

}
variable "services_flavor" {
  description = "Flavor for the Services node (CPU VM)"
  type        = string
  default     = "m1.large"
}


variable "gpu_image" {
  description = "Image name for GPU node (CUDA-enabled)"
  type        = string
  default     = "CC-Ubuntu24.04-CUDA"
}
variable "gpu_flavor" {
  description = "Flavor for the GPU node"
  type        = string
  default     = "baremetal"  # Chameleon uses "baremetal" flavor for any reserved baremetal node&#8203;:contentReference[oaicite:9]{index=9}


# SSH Key Pair
variable "keypair_name" {
  description = "SSH keypair name registered in OpenStack"
  type        = string
  default     = "mlops_proj_key"
}
variable "public_key_path" {
  description = "Path to the local public SSH key file"
  type        = string
  default     = "~/.ssh/mlops_proj_key.pub"
}

# Private network
variable "network_name" {
  description = "Name of the private network for inter-node communication"
  type        = string
  default     = "mlops-net"
}
variable "network_cidr" {
  description = "CIDR block for the private subnet"
  type        = string
  default     = "10.0.0.0/24"
}

# Security group
variable "security_group_name" {
  description = "Name for the security group"
  type        = string
  default     = "mlops-secgrp"
}

# Optional staging node
variable "enable_staging" {
  description = "Whether to provision a staging node"
  type        = bool
  default     = false
}
variable "staging_image" {
  description = "Image name for staging node"
  type        = string
  default     = "CC-Ubuntu24.04"
}
variable "staging_flavor" {
  description = "Flavor for staging node"
  type        = string
  default     = "m1.small"
}