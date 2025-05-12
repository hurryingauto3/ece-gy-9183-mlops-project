#!/bin/bash

# run_terraform.sh
# Executes the Terraform workflow to provision VMs on Chameleon.

# Exit immediately if a command exits with a non-zero status.
set -e
# Print commands and their arguments as they are executed.
set -x

# --- Configuration ---
TERRAFORM_DIR="./devops/terraform"
TERRAFORM_PLAN_FILE="tfplan"
# SSH_KEY_NAME="my-agri-mlops-key" # <<<< Update this in your variables.tf file instead

# You MUST source your openrc.sh file before running this script.
# This script checks if required OS_* environment variables are set.

# --- Functions ---
check_openstack_creds() {
    echo "Checking OpenStack environment variables..."
    # Basic checks for essential credentials. Add more if your setup requires them.
    if [ -z "$OS_AUTH_URL" ] || [ -z "$OS_USERNAME" ] || [ -z "$OS_PASSWORD" ] || [ -z "$OS_PROJECT_NAME" ]; then
        echo "Error: OpenStack environment variables are not set."
        echo "Please source your Chameleon openrc.sh file before running this script."
        exit 1
    fi
    echo "OpenStack credentials seem to be set."
}

check_ssh_key() {
    local key_name
    # Attempt to read key name from variables.tf if possible, otherwise use a default/known name
    # More robust way would parse terraform output or variables.tf
    # For simplicity, let's rely on the variable defined in variables.tf being correct
    # and assume its name is consistent. The terraform apply will fail if the key_pair variable is wrong.
    # We can just check if the local file exists based on the *expected* name.
    key_name=$(grep 'variable "chameleon_keypair_name"' "$TERRAFORM_DIR/variables.tf" | grep 'default' | cut -d'"' -f 2)
    if [ -z "$key_name" ]; then
        echo "Warning: Could not automatically determine SSH key name from variables.tf. Assuming default 'my-agri-mlops-key'."
        key_name="my-agri-mlops-key" # Fallback
    fi

    local ssh_key_path="$HOME/.ssh/${key_name}.pem"
    echo "Checking for SSH private key at: $ssh_key_path"
    if [ ! -f "$ssh_key_path" ]; then
        echo "Error: SSH private key not found at $ssh_key_path."
        echo "Please ensure you have created the keypair '$key_name' in Chameleon and downloaded the private key to this location."
        echo "Update the 'chameleon_keypair_name' variable in $TERRAFORM_DIR/variables.tf if the name is different."
        exit 1
    fi
    # Ensure correct permissions for the private key (important for SSH)
    chmod 600 "$ssh_key_path"
    echo "SSH private key found and permissions set."
}


# --- Main Execution ---
echo "--- Starting Terraform Provisioning ---"

# Check for OpenStack credentials
check_openstack_creds

# Navigate to the Terraform directory
echo "Navigating to Terraform directory: $TERRAFORM_DIR"
if [ ! -d "$TERRAFORM_DIR" ]; then
    echo "Error: Terraform directory not found at $TERRAFORM_DIR"
    exit 1
fi
cd "$TERRAFORM_DIR"

# Check for the SSH private key locally
check_ssh_key

# 1. Initialize Terraform
echo "Initializing Terraform..."
# -input=false prevents interactive prompts for variables if not set via env/file
terraform init -input=false

# 2. Plan the Terraform changes
echo "Planning Terraform changes..."
# -input=false prevents interactive prompts
terraform plan -out="$TERRAFORM_PLAN_FILE" -input=false

# 3. Apply the Terraform changes
echo "Applying Terraform changes..."
# -auto-approve skips confirmation. REMOVE THIS FOR MANUAL APPROVAL IN PRODUCTION.
# -input=false prevents interactive prompts
terraform apply -auto-approve "$TERRAFORM_PLAN_FILE" -input=false

# Check the exit status of terraform apply
if [ $? -ne 0 ]; then
    echo "--- Terraform Apply Failed ---"
    # Terraform errors are usually descriptive.
    exit 1
fi

echo "--- Terraform Provisioning Complete ---"

# The local-exec provisioner in main.tf should have generated the Ansible inventory file.
# Output the generated inventory path
INVENTORY_PATH="../ansible/inventory.ini"
if [ -f "$INVENTORY_PATH" ]; then
    echo "Generated Ansible inventory at: $INVENTORY_PATH"
    echo "Contents:"
    cat "$INVENTORY_PATH"
else
     echo "Warning: Ansible inventory file was not found after terraform apply at $INVENTORY_PATH."
     echo "Please check your main.tf and terraform apply output for errors in the local-exec provisioner."
fi


echo "Proceed to run the Ansible setup script."

exit 0