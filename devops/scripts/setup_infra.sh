#!/usr/bin/env bash
#
# setup_infra.sh
#
# Script that creates basic infrastructure on Chameleon (KVM@TACC)
# for the class project. It follows set naming conventions and guidelines.
#
# Usage: ./setup_infra.sh <PROJECT_NUMBER> <SHARED_KEY_NAME> <DATASET_NAME>
#
# Example: ./setup_infra.sh 0 project0_key dataset1
#
# Assumptions:
#  - You have sourced your OpenStack credentials (openrc or clouds.yaml).
#  - The shared SSH key <SHARED_KEY_NAME> is already uploaded to KVM@TACC.
#  - You have the OpenStack CLI and swift client installed.
#  - You have an appropriate image and flavor name for your VMs.

# Exit on any error
set -e

# Basic usage check
if [ $# -lt 3 ]; then
    echo "Usage: $0 <PROJECT_NUMBER> <SHARED_KEY_NAME> <DATASET_NAME>"
    exit 1
fi

PROJECT_NUM=$4
SHARED_KEY_NAME=$project4_key.pem
DATASET_NAME=$CropNet

# ----------------------
# 1. Define Resource Names (Suffix: -project${PROJECT_NUM})
# ----------------------
NETWORK_NAME="net-project${PROJECT_NUM}"
SUBNET_NAME="subnet-project${PROJECT_NUM}"
ROUTER_NAME="router-project${PROJECT_NUM}"
SECURITY_GROUP_NAME="secgroup-project${PROJECT_NUM}"
VOLUME_NAME="volume-project${PROJECT_NUM}"
CONTAINER_NAME="${DATASET_NAME}-project${PROJECT_NUM}"
VM_NAME="vm1-project${PROJECT_NUM}"

# Adjust these as needed:
VM_IMAGE="CC-Ubuntu20.04"   # preferred image name
VM_FLAVOR="m1.medium"       # might switch to a larger VM on KVM@TACC if needed
SUBNET_CIDR="192.168.100.0/24"  # subnet range used in assignment 1
GW_NETWORK="public"         # name of external network for router gateway

# ----------------------
# 2. Create Network, Subnet, Router
# ----------------------
echo "Creating network: $NETWORK_NAME ..."
openstack network create "$NETWORK_NAME"

echo "Creating subnet: $SUBNET_NAME ..."
openstack subnet create "$SUBNET_NAME" \
  --network "$NETWORK_NAME" \
  --subnet-range "$SUBNET_CIDR"

echo "Creating router: $ROUTER_NAME ..."
openstack router create "$ROUTER_NAME"

echo "Setting router gateway to $GW_NETWORK ..."
openstack router set "$ROUTER_NAME" --external-gateway "$GW_NETWORK"

echo "Attaching subnet to router ..."
openstack router add subnet "$ROUTER_NAME" "$SUBNET_NAME"

# ----------------------
# 3. Create a Security Group
# ----------------------
echo "Creating security group: $SECURITY_GROUP_NAME ..."
openstack security group create "$SECURITY_GROUP_NAME" --description "Security group for project${PROJECT_NUM}"

# Allow SSH (port 22) and ICMP (ping) from anywhere
openstack security group rule create --proto tcp --dst-port 22 "$SECURITY_GROUP_NAME"
openstack security group rule create --proto icmp "$SECURITY_GROUP_NAME"

# ----------------------
# 4. Create a Persistent Volume (Optional)
# ----------------------
# You can skip this if you don’t need block storage. Adjust size as necessary.
echo "Creating volume: $VOLUME_NAME ..."
openstack volume create --size 10 "$VOLUME_NAME"

# ----------------------
# 5. Create an Object Storage Container (for your dataset)
# ----------------------
echo "Creating object storage container: $CONTAINER_NAME ..."
openstack container create "$CONTAINER_NAME"

# You can upload data to it using:
#   openstack object create "$CONTAINER_NAME" <filename_or_folder>

# ----------------------
# 6. Launch a VM
# ----------------------
echo "Launching VM: $VM_NAME ..."
openstack server create "$VM_NAME" \
  --image "$VM_IMAGE" \
  --flavor "$VM_FLAVOR" \
  --key-name "$SHARED_KEY_NAME" \
  --network "$NETWORK_NAME" \
  --security-group "$SECURITY_GROUP_NAME"

# Wait for VM to be ACTIVE
echo "Waiting for VM to become ACTIVE ..."
openstack server wait "$VM_NAME" --status ACTIVE

# ----------------------
# 7. Allocate/Attach a Floating IP (only 1 at a time, per site!)
# ----------------------
echo "Allocating floating IP ..."
FLOATING_IP=$(openstack floating ip create "$GW_NETWORK" -f value -c floating_ip_address)

echo "Associating floating IP $FLOATING_IP with $VM_NAME ..."
openstack server add floating ip "$VM_NAME" "$FLOATING_IP"

# Output the floating IP so you can SSH
echo "--------------------------------------------------------------"
echo "All done! You can now SSH to your VM using:"
echo "   ssh -i <YOUR_PRIVATE_KEY> ubuntu@${FLOATING_IP}"
echo "--------------------------------------------------------------"


# ----------------------
# (TODO when training) 8. Reserve Bare Metal (GPU) Nodes
# ----------------------
# When we need GPU bare metal at CHI@UC or CHI@TACC, we would do something like:
#
# openstack lease create lease-gpu-project${PROJECT_NUM} \
#   --start-date <start_time> \
#   --end-date <end_time> \
#   --resource "physres_name=gpu_rtx6000, resource_type=physical:host, min=1, max=1"
#
# Then when it’s ACTIVE, launch a bare metal instance on it.
#
# OR, use the Horizon dashboard to reserve a bare metal node.
#   - Go to "Compute" -> "Bare Metal" -> "Lease"
#   - Fill in the form with the resource name and time range.
#   - Click "Create Lease"
#   - Wait for the lease to be ACTIVE.
#   - Go to "Compute" -> "Bare Metal" -> "Instances"
#   - Click "Launch Instance"
# ----------------------

