# devops/terraform/main.tf

# Configure the OpenStack provider
# This provider uses environment variables (like OS_AUTH_URL, OS_USERNAME, etc.)
# which are typically set by sourcing the project's openrc.sh file.
terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.50" # Use a recent compatible version
    }
  }

  # Optional: Configure a backend for storing Terraform state.
  # Using a remote backend (like Swift, S3, or a dedicated Terraform backend)
  # is recommended for collaboration and robustness in production.
  # For a simple start, you can use the default local backend (terraform.tfstate).
  # backend "swift" {
  #   container = "terraform-state" # Create this container in Swift
  #   region    = var.chameleon_region_name # Use region from your variables or env vars
  #   # Other OS_* credentials typically read from environment
  # }
}

# Configure the OpenStack provider instance
provider "openstack" {
  # The provider will automatically read authentication details from OS_* environment variables.
  # You can explicitly set the region name here if needed, matching your reservation.
  # region = var.chameleon_region_name # Example if you defined a region variable
}

# Look up the network ID by name
# This is needed to attach the VMs and the floating IP to the correct network.
data "openstack_networking_network_v2" "network" {
  name = var.chameleon_network_name
}

# Create a Security Group for the MLops VMs
# This group will define which ports are open for incoming traffic.
resource "openstack_networking_secgroup_v2" "mlops_sg" {
  name        = "${var.vm_name_prefix}-sg"
  description = "Security group for Agri MLops VMs on Chameleon"
}

# Add rules to the Security Group to allow incoming TCP traffic on specified ports
# WARNING: remote_ip_prefix = "0.0.0.0/0" allows access from ANYWHERE.
# This is convenient for testing but highly insecure for production.
# Restrict this to known IPs or specific subnets in a real deployment.
resource "openstack_networking_secgroup_rule_v2" "mlops_sg_rules_tcp" {
  count             = length(var.ports_to_open) # Create one rule for each port
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = var.ports_to_open[count.index]
  port_range_max    = var.ports_to_open[count.index]
  remote_ip_prefix  = "0.0.0.0/0" # !! INSECURE FOR PRODUCTION !!
  security_group_id = openstack_networking_secgroup_v2.mlops_sg.id
  description       = "Allow incoming TCP traffic on port ${var.ports_to_open[count.index]}"
}

# (Optional) Add a rule to allow ICMP (ping) for basic connectivity tests
resource "openstack_networking_secgroup_rule_v2" "mlops_sg_rule_icmp" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "icmp"
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.mlops_sg.id
  description       = "Allow incoming ICMP traffic"
}


# --- Create the CPU VM Instance ---
# This VM will host the Feature Service, Model Serving, MLflow, Prometheus, Grafana.
resource "openstack_compute_instance_v2" "mlops_cpu_vm" {
  name              = "${var.vm_name_prefix}-cpu-vm-${formatdate("YYYYMMDDhhmmss", timestamp())}" # Unique name
  image_name        = var.chameleon_image_name
  flavor_name       = var.cpu_flavor_name # Use the CPU flavor variable
  key_pair          = var.chameleon_keypair_name
  security_groups   = [openstack_networking_secgroup_v2.mlops_sg.name] # Associate the security group

  networks {
    uuid = data.openstack_networking_network_v2.network.id # Attach to the specified network
  }

  # Provisioner to wait for SSH access and cloud-init to finish
  # This ensures the VM is ready for Ansible connection after Terraform applies it.
  connection {
    type        = "ssh"
    # Use the private IP for connection initially within the internal network
    host        = self.access_ip_v4 # Terraform often uses access_ip_v4 which might be floating or fixed depending on setup/pool
    user        = "cc" # Default user for CC Ubuntu images (adjust if using a different image)
    # IMPORTANT: Path to your local private key file used for SSH
    private_key = file("~/.ssh/${var.chameleon_keypair_name}.pem")
    timeout     = "5m" # Wait up to 5 minutes for SSH to become available
    # You might need to explicitly set host if access_ip_v4 isn't the private IP you expect
    # host = openstack_networking_network_v2.network.fixed_ip_v4
  }

  # Run a command on the VM after SSH is ready to wait for cloud-init completion
  provisioner "remote-exec" {
    inline = [
      "cloud-init status --wait > /dev/null", # Wait for cloud-init to finish
      "sudo apt-get update", # Example command: update package list
      # Add other basic setup commands if needed before Ansible takes over
    ]
  }
}

# --- Create the GPU VM Instance ---
# This VM will primarily run the Model Training jobs.
resource "openstack_compute_instance_v2" "mlops_gpu_vm" {
  name              = "${var.vm_name_prefix}-gpu-vm-${formatdate("YYYYMMDDhhmmss", timestamp())}" # Unique name
  image_name        = var.chameleon_image_name
  flavor_name       = var.gpu_flavor_name # Use the GPU flavor variable
  key_pair          = var.chameleon_keypair_name
  security_groups   = [openstack_networking_secgroup_v2.mlops_sg.name] # Associate the security group

  networks {
    uuid = data.openstack_networking_network_v2.network.id # Attach to the specified network
  }

  # Provisioner to wait for SSH access and cloud-init to finish
  connection {
    type        = "ssh"
    host        = self.access_ip_v4 # Use the assigned IP
    user        = "cc" # Default user for CC Ubuntu images
    private_key = file("~/.ssh/${var.chameleon_keypair_name}.pem") # Path to your local private key
    timeout     = "5m" # Wait up to 5 minutes
  }

  # Run a command on the VM after SSH is ready
  provisioner "remote-exec" {
    inline = [
      "cloud-init status --wait > /dev/null", # Wait for cloud-init
      "sudo apt-get update", # Example command
    ]
  }
}

# --- Create a Floating IP ---
# This is needed to access services (MLflow UI, Model Serving API, Grafana, Prometheus)
# from the public internet. Associate it with the CPU VM.
resource "openstack_networking_floatingip_v2" "mlops_fip" {
  # Optional: Specify a floating IP pool if your project has multiple
  # pool = "external" # Example pool name, confirm in Chameleon UI
  network_id = data.openstack_networking_network_v2.network.id # Associate with the same network
}

# Associate the Floating IP with the CPU VM Instance
resource "openstack_compute_floatingip_associate_v2" "mlops_fip_associate" {
  floating_ip = openstack_networking_floatingip_v2.mlops_fip.address
  instance_id = openstack_compute_instance_v2.mlops_cpu_vm.id # Associate with the CPU VM
}

# --- Optional: Floating IP for GPU VM ---
# Uncomment this block if you also want a public IP for the GPU VM (e.g., for direct SSH debugging)
# resource "openstack_networking_floatingip_v2" "mlops_gpu_fip" {
#   network_id = data.openstack_networking_network_v2.network.id
# }
#
# resource "openstack_compute_floatingip_associate_v2" "mlops_gpu_fip_associate" {
#   floating_ip = openstack_networking_floatingip_v2.mlops_gpu_fip.address
#   instance_id = openstack_compute_instance_v2.mlops_gpu_vm.id
# }


# --- Output Information ---
# Output variables help retrieve created resource attributes after apply.
# These are crucial for passing VM IPs to Ansible.

output "cpu_vm_name" {
  description = "Name of the created CPU VM instance"
  value       = openstack_compute_instance_v2.mlops_cpu_vm.name
}

output "cpu_vm_id" {
  description = "ID of the created CPU VM instance"
  value       = openstack_compute_instance_v2.mlops_cpu_vm.id
}

output "cpu_vm_private_ip" {
  description = "Private IP address of the CPU VM instance on the internal network"
  # Access the first network's first fixed IP
  value       = openstack_compute_instance_v2.mlops_cpu_vm.network[0].fixed_ip_v4
}

output "cpu_vm_floating_ip" {
  description = "Floating (public) IP address associated with the CPU VM"
  value       = openstack_networking_floatingip_v2.mlops_fip.address
}

output "cpu_ssh_command" {
  description = "SSH command to connect to the CPU VM using the floating IP"
  value       = "ssh -i ~/.ssh/${var.chameleon_keypair_name}.pem cc@${openstack_networking_floatingip_v2.mlops_fip.address}"
}


output "gpu_vm_name" {
  description = "Name of the created GPU VM instance"
  value       = openstack_compute_instance_v2.mlops_gpu_vm.name
}

output "gpu_vm_id" {
  description = "ID of the created GPU VM instance"
  value       = openstack_compute_instance_v2.mlops_gpu_vm.id
}

output "gpu_vm_private_ip" {
  description = "Private IP address of the GPU VM instance on the internal network"
  value       = openstack_compute_instance_v2.mlops_gpu_vm.network[0].fixed_ip_v4
}

# Optional: Output GPU VM Floating IP if created
# output "gpu_vm_floating_ip" {
#   description = "Floating (public) IP address associated with the GPU VM"
#   value       = openstack_networking_floatingip_v2.mlops_gpu_fip.address
# }

# Output Ansible Inventory structure (optional, alternative to manual inventory)
# This writes a basic inventory file using a local-exec provisioner after VMs are ready.
# Assumes Ansible inventory file is in devops/ansible/inventory.ini relative to this file.
resource "null_resource" "ansible_inventory_generator" {
  # This depends on both VMs being created and SSHable
  depends_on = [openstack_compute_instance_v2.mlops_cpu_vm, openstack_compute_instance_v2.mlops_gpu_vm]

  provisioner "local-exec" {
    # Command runs on the machine where `terraform apply` is executed
    command = <<EOT
      # Define path to the inventory file relative to where this command runs
      INV_FILE="../ansible/inventory.ini"

      # Remove old inventory file if it exists
      if [ -f "$INV_FILE" ]; then
          rm "$INV_FILE"
      fi

      echo "[cpu_vms]" >> "$INV_FILE"
      echo "${openstack_compute_instance_v2.mlops_cpu_vm.network[0].fixed_ip_v4} ansible_user=cc ansible_ssh_private_key_file=~/.ssh/${var.chameleon_keypair_name}.pem" >> "$INV_FILE"
      echo "" >> "$INV_FILE"

      echo "[gpu_vms]" >> "$INV_FILE"
      echo "${openstack_compute_instance_v2.mlops_gpu_vm.network[0].fixed_ip_v4} ansible_user=cc ansible_ssh_private_key_file=~/.ssh/${var.chameleon_keypair_name}.pem" >> "$INV_FILE"
      echo "" >> "$INV_FILE"

      echo "[all:vars]" >> "$INV_FILE"
      echo "project_remote_path=/home/cc/agri-mlops" >> "$INV_FILE"
      echo "ansible_python_interpreter=/usr/bin/python3" # Specify python3 path

      echo "Generated Ansible inventory: $INV_FILE"
      cat "$INV_FILE"

    EOT
    # Ensure the local-exec runs from the directory where terraform apply is executed
    # working_dir = path.module # By default, it runs from the module directory (devops/terraform)
  }
}