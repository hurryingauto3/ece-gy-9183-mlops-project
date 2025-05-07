# devops/terraform/variables.tf

variable "chameleon_image_name" {
  description = "Name of the Chameleon VM image to use (e.g., CC-Ubuntu20.04)"
  type        = string
  default     = "CC-Ubuntu20.04" # Use a common, stable Ubuntu image
}

variable "cpu_flavor_name" {
  description = "Name of the Chameleon VM flavor (size) for the CPU VM (e.g., m1.medium)"
  type        = string
  default     = "m1.medium" # Default CPU size
}

variable "gpu_flavor_name" {
  description = "Name of the Chameleon VM flavor (size) for the GPU VM (e.g., gpu1.small)"
  type        = string
  # YOU MUST CHANGE THIS DEFAULT TO YOUR RESERVED GPU FLAVOR NAME
  default = "gpu1.small" # Example GPU flavor name - VERIFY THIS IN CHAMELEON UI
}


variable "chameleon_keypair_name" {
  description = "Name of the SSH keypair registered in Chameleon"
  type        = string
  # YOU MUST CHANGE THIS DEFAULT to the name of your SSH keypair
  default = "my-agri-mlops-key"
}

variable "chameleon_network_name" {
  description = "Name of the network to attach the VMs to (e.g., sharednet1)"
  type        = string
  # YOU MUST CHANGE THIS DEFAULT to your project's network name
  default = "sharednet1" # Common default, verify in Chameleon UI/openstack CLI
}

variable "vm_name_prefix" {
  description = "Prefix for the VM names"
  type        = string
  default     = "agri-mlops"
}

variable "ports_to_open" {
  description = "List of TCP ports to open in the security group (SSH, MLflow, APIs, Monitoring UIs)"
  type        = list(number)
  default = [
    22,   # SSH - Required for Ansible
    5000, # MLflow UI/Tracking
    8000, # Model Serving API
    8001, # Feature Service API (opening externally for testing/direct access)
    9090, # Prometheus UI
    3000  # Grafana UI
  ]
}

# Optional: Variable for the region name if needed by provider or backend
# variable "chameleon_region_name" {
#   description = "Chameleon Cloud region name"
#   type        = string
#   default     = "kvm" # Example region name, verify in Chameleon UI
# }