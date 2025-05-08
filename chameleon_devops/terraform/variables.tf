# ──────────────────────────────────────────
# OpenStack / Chameleon settings
# ──────────────────────────────────────────
variable "cloud_kvm" {
  description = "clouds.yaml entry for KVM@TACC"
  type        = string
  default     = "kvm"
}
variable "services_region" {
  description = "Region for the Services node (KVM@TACC)"
  type        = string
  default     = "KVM@TACC"
}
variable "cloud_chi" {
  description = "clouds.yaml entry for CHI@TACC"
  type        = string
  default     = "chi"
}
variable "chi_region" {
  description = "Region for the GPU node (CHI@TACC)"
  type        = string
  default     = "CHI@TACC"
}
variable "cloud_uc" {
  description = "clouds.yaml entry for CHI@UC"
  type        = string
  default     = "uc"
}
variable "uc_region" {
  description = "Region for UC@CHI"
  type        = string
  default     = "CHI@UC"
}

variable "user_tag" {
  description = "Your NetID or initials to avoid name collisions"
  type        = string
}

# ──────────────────────────────────────────
# Networking
# ──────────────────────────────────────────
variable "ext_net_name" {
  description = "Name of the external/public network for floating IPs"
  type        = string
  default     = "public"
}
variable "network_name" {
  description = "Name of the private network"
  type        = string
  default     = "mlops-net"
}
variable "network_cidr" {
  description = "CIDR block for the private subnet"
  type        = string
  default     = "10.0.0.0/24"
}
variable "network_gateway" {
  description = "Gateway IP for the private subnet"
  type        = string
  default     = "10.0.0.1"
}
variable "network_pool_start" {
  description = "First IP in allocation pool"
  type        = string
  default     = "10.0.0.50"
}
variable "network_pool_end" {
  description = "Last IP in allocation pool"
  type        = string
  default     = "10.0.0.200"
}
variable "dns_nameservers" {
  description = "DNS servers for the subnet"
  type        = list(string)
  default     = ["8.8.8.8", "8.8.4.4"]
}

# ──────────────────────────────────────────
# VM images & flavors
# ──────────────────────────────────────────
variable "services_image" {
  description = "Image name for Services node"
  type        = string
  default     = "CC-Ubuntu24.04"
}
variable "services_flavor" {
  description = "Flavor for the Services node"
  type        = string
  default     = "m1.large"
}
variable "gpu_image" {
  description = "Image name for GPU node (CUDA‑enabled)"
  type        = string
  default     = "CC-Ubuntu24.04-CUDA"
}
variable "gpu_image_id" {
  description = "Glance UUID of the CUDA‑enabled GPU image"
  type        = string
}
variable "gpu_flavor" {
  description = "Flavor for the GPU node"
  type        = string
  default     = "baremetal"
}

# ──────────────────────────────────────────
# SSH & Security
# ──────────────────────────────────────────
variable "keypair_name" {
  description = "SSH keypair name"
  type        = string
  default     = "mlops_proj_key"
}
variable "public_key_path" {
  description = "Path to public SSH key file"
  type        = string
  default     = "~/.ssh/mlops_proj_key.pub"
}
variable "security_group_name" {
  description = "Base name for the security group"
  type        = string
  default     = "mlops-secgrp"
}

# ──────────────────────────────────────────
# Optional staging
# ──────────────────────────────────────────
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

# ──────────────────────────────────────────
# GPU reservation
# ──────────────────────────────────────────
variable "gpu_reservation_id" {
  description = "Chameleon lease reservation UUID for the GPU host"
  type        = string
  default     = ""
}