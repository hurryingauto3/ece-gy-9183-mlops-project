# cloud = "chameleon"
cloud = "kvm"

cloud_kvm       = "kvm"
services_region = "KVM@TACC"
cloud_chi       = "chi"
chi_region      = "CHI@TACC"
gpu_region      = "CHI@TACC"
cloud_uc        = "uc"
uc_region       = "CHI@UC"

ext_net_name = "public"

# other assignments…
# ext_net_name        = "public-ext-net"
services_image      = "CC-Ubuntu24.04"
services_flavor     = "m1.large"
gpu_image           = "CC-Ubuntu24.04-CUDA"
gpu_flavor          = "baremetal"
keypair_name        = "mlops_proj_key"
public_key_path     = "~/.ssh/mlops_proj_key.pub"
network_name        = "mlops-net"
network_cidr        = "10.0.0.0/24"
security_group_name = "mlops-secgrp"
enable_staging      = false

# Assign your reservation UUID here:
# gpu_reservation_id = "c790c6bc-e2b5-4267-87b9-f55296fd16a5"
# gpu_reservation_id = "c320853a-01c2-430d-8068-885c91a15926"
# gpu_reservation_id = "47d0e0cd-883c-4639-a646-a47fde6e7c4a"
# gpu_reservation_id = "8c7169a8-0f36-4d4a-a1f3-eaddf7646112"
# gpu_reservation_id = "15b16b92-95c9-4d4e-87b9-5e2eac70ee29"

gpu_reservation_id = "d97ab437-5cab-47a6-99dc-001ba9822c18"

gpu_image_id_uc = "45661d6e-d442-48b2-892f-e39a246011cc"
# gpu_reservation_id_uc = "d97ab437-5cab-47a6-99dc-001ba9822c18"
# gpu_reservation_id_uc = "1def9c07-1eda-44fc-9564-dae33761ba88"

# gpu_reservation_id_uc = "dfac712e-3045-4a19-ad2c-7ff0a0ba8d45"
gpu_reservation_id_uc = "a320ee25-a716-48c5-b880-80ec7e32c56b"

