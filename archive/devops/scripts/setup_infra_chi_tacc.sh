#!/usr/bin/env bash

# setup_infra_chi_tacc.sh
# Idempotent script to set up basic infrastructure for a project on CHI@TACC.

# Usage: ./setup_infra_chi_tacc.sh <project_number> <key_name> <dataset_name>
# Example: ./setup_infra_chi_tacc.sh 4 project4_key CropNet

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <project_number> <key_name> <dataset_name>"
    exit 1
fi

PROJECT_NUM="$1"
SHARED_KEY_NAME="$2"
DATASET_NAME="$3"

# ----------------------------
# 1. Define Names
# ----------------------------
NETWORK_NAME="net-project${PROJECT_NUM}"
SUBNET_NAME="subnet-project${PROJECT_NUM}"
ROUTER_NAME="router-project${PROJECT_NUM}"
SECGRP_NAME="secgroup-project${PROJECT_NUM}"
VOLUME_NAME="volume-project${PROJECT_NUM}"
CONTAINER_NAME="${DATASET_NAME}-project${PROJECT_NUM}"
VM_NAME="vm1-project${PROJECT_NUM}"

# Adjust these for your needs:
VM_IMAGE="CC-Ubuntu20.04"
# VM_FLAVOR="m1.medium"
VM_FLAVOR="baremetal"
SUBNET_CIDR="192.168.100.0/24"
EXTERNAL_NETWORK="public" # The external network name on CHI@TACC
VOLUME_SIZE=20 # Size in GB for your volume

echo "========== Idempotent Setup for Project ${PROJECT_NUM} on CHI@TACC =========="
echo "This will create or skip resources that already exist."
echo

# ----------------------------
# 2. Create Network (if not exists)
# ----------------------------
echo "[+] Checking network: ${NETWORK_NAME}"
if ! openstack network show "$NETWORK_NAME" &>/dev/null; then
    echo " -> Creating network ${NETWORK_NAME}..."
    openstack network create "$NETWORK_NAME"
else
    echo " -> Network ${NETWORK_NAME} already exists. Skipping."
fi

# ----------------------------
# 3. Create Subnet (if not exists)
# ----------------------------
echo "[+] Checking subnet: ${SUBNET_NAME}"
if ! openstack subnet show "$SUBNET_NAME" &>/dev/null; then
    echo " -> Creating subnet ${SUBNET_NAME}..."
    openstack subnet create "$SUBNET_NAME" \
        --network "$NETWORK_NAME" \
        --subnet-range "$SUBNET_CIDR"
else
    echo " -> Subnet ${SUBNET_NAME} already exists. Skipping."
fi

# ----------------------------
# 4. Create Router (if not exists) and attach subnet
# ----------------------------
echo "[+] Checking router: ${ROUTER_NAME}"
if ! openstack router show "$ROUTER_NAME" &>/dev/null; then
    echo " -> Creating router ${ROUTER_NAME}..."
    openstack router create "$ROUTER_NAME"
    echo " -> Setting router gateway to ${EXTERNAL_NETWORK}..."
    openstack router set "$ROUTER_NAME" --external-gateway "$EXTERNAL_NETWORK"
    echo " -> Attaching subnet ${SUBNET_NAME} to router..."
    openstack router add subnet "$ROUTER_NAME" "$SUBNET_NAME"
else
    echo " -> Router ${ROUTER_NAME} already exists. Ensuring gateway/subnet attached..."
    # If external gateway not set or changed, set it
    CURRENT_GATEWAY="$(openstack router show "$ROUTER_NAME" -f value -c external_gateway_info || true)"
    if [[ "$CURRENT_GATEWAY" == "null" ]]; then
        openstack router set "$ROUTER_NAME" --external-gateway "$EXTERNAL_NETWORK"
    fi
    
    # Attach the subnet if not attached
    SUBNETS_ATTACHED="$(openstack router show "$ROUTER_NAME" -f json | jq -r '.interfaces_info[].subnet_id' || true)"
    if ! echo "$SUBNETS_ATTACHED" | grep -q "$(openstack subnet show "$SUBNET_NAME" -f value -c id)"; then
        openstack router add subnet "$ROUTER_NAME" "$SUBNET_NAME"
    fi
fi

# ----------------------------
# 5. Create Security Group (if not exists)
# ----------------------------
echo "[+] Checking security group: ${SECGRP_NAME}"
if ! openstack security group show "$SECGRP_NAME" &>/dev/null; then
    echo " -> Creating security group ${SECGRP_NAME}"
    openstack security group create "$SECGRP_NAME" --description "Security group for project${PROJECT_NUM}"
    # Allow SSH (port 22) from anywhere
    openstack security group rule create --proto tcp --dst-port 22 "$SECGRP_NAME"
    # Allow ping (ICMP)
    openstack security group rule create --proto icmp "$SECGRP_NAME"
else
    echo " -> Security group ${SECGRP_NAME} already exists. Skipping."
fi

# # ----------------------------
# # 6. Create a Persistent Volume (if not exists)
# # ----------------------------
# echo "[+] Checking volume: ${VOLUME_NAME}"
# if ! openstack volume show "$VOLUME_NAME" &>/dev/null; then
#     echo " -> Creating volume ${VOLUME_NAME} (${VOLUME_SIZE}GB)"
#     openstack volume create --size "$VOLUME_SIZE" "$VOLUME_NAME"
# else
#     echo " -> Volume ${VOLUME_NAME} already exists. Skipping."
# fi

# # ----------------------------
# # 7. Create an Object Storage Container (if not exists)
# # ----------------------------
# echo "[+] Checking object storage container: ${CONTAINER_NAME}"
# if ! openstack container show "$CONTAINER_NAME" &>/dev/null; then
#     echo " -> Creating container ${CONTAINER_NAME}..."
#     openstack container create "$CONTAINER_NAME"
# else
#     echo " -> Container ${CONTAINER_NAME} already exists. Skipping."
# fi

# ----------------------------
# 6. Create a Persistent Volume (if available and not exists)
# ----------------------------
echo "[+] Checking volume: ${VOLUME_NAME}"
if openstack volume create --size "$VOLUME_SIZE" "$VOLUME_NAME" &>/dev/null; then
    echo " -> Volume ${VOLUME_NAME} created successfully"
elif openstack volume show "$VOLUME_NAME" &>/dev/null; then
    echo " -> Volume ${VOLUME_NAME} already exists. Skipping."
else
    echo " -> WARNING: Could not create or verify volume. Block storage service may not be available."
    echo " -> Will continue with the rest of the setup."
fi

# ----------------------------
# 7. Create an Object Storage Container (if available and not exists)
# ----------------------------
echo "[+] Checking object storage container: ${CONTAINER_NAME}"
if openstack container create "${CONTAINER_NAME}" &>/dev/null; then
    echo " -> Container ${CONTAINER_NAME} created successfully"
elif openstack container show "${CONTAINER_NAME}" &>/dev/null; then
    echo " -> Container ${CONTAINER_NAME} already exists. Skipping."
else
    echo " -> WARNING: Could not create or verify container. Object storage may not be available."
    echo " -> Will continue with the rest of the setup."
fi


# Do this when the reservation is valid

# # ----------------------------
# # 8. Launch a VM (if not exists)
# # ----------------------------
# echo "[+] Checking VM: ${VM_NAME}"
# if ! openstack server show "$VM_NAME" &>/dev/null; then
#     echo " -> Creating VM ${VM_NAME}..."
#     openstack server create "$VM_NAME" \
#         --image "$VM_IMAGE" \
#         --flavor "$VM_FLAVOR" \
#         --key-name "$SHARED_KEY_NAME" \
#         --network "$NETWORK_NAME" \
#         --security-group "$SECGRP_NAME"
#     echo " -> Waiting for VM to become ACTIVE..."
#     openstack server wait "$VM_NAME" --status ACTIVE --timeout 300
# else
#     echo " -> VM ${VM_NAME} already exists. Skipping creation."
# fi

# # ----------------------------
# # 9. Allocate & Associate Floating IP if not already assigned
# # ----------------------------
# echo "[+] Ensuring ${VM_NAME} has a floating IP..."
# EXISTING_IPS="$(openstack server show "${VM_NAME}" -f json | jq -r '.addresses' || true)"
# if echo "${EXISTING_IPS}" | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' >/dev/null; then
#     # VM already has a floating IP
#     CURRENT_IP="$(echo "${EXISTING_IPS}" | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -n 1)"
#     echo " -> VM ${VM_NAME} seems to already have a floating IP: ${CURRENT_IP}"
# else
#     # Allocate new floating IP
#     echo " -> Allocating a new floating IP from ${EXTERNAL_NETWORK}..."
#     CURRENT_IP=$(openstack floating ip create "${EXTERNAL_NETWORK}" -f value -c floating_ip_address)
#     echo " -> Attaching floating IP ${CURRENT_IP} to ${VM_NAME}"
#     openstack server add floating ip "${VM_NAME}" "${CURRENT_IP}"
#     echo " -> Done. VM now accessible at ${CURRENT_IP}"
# fi

echo
echo "============================================================"
echo "Idempotent setup complete for project${PROJECT_NUM}!"
echo "Resources created or found (no duplicates)."
echo
openstack server list --name "$VM_NAME"

# Print a direct SSH command
echo
echo "====================================================================="
echo "Copy/paste the following command to SSH into your VM:"
echo "ssh -i ~/.ssh/project4_key.pem ubuntu@${CURRENT_IP}"
echo "====================================================================="
