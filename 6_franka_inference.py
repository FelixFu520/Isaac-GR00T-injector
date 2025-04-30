import os
import torch
import gr00t

from gr00t.data.dataset import LiberoSingleDataset
from gr00t.model.policy import Gr00tPolicy

# change the following paths
MODEL_PATH = "/root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/output/6_franka_libero_object_no_noops_lerobot_20000"

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "/root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/data/IPEC-COMMUNITY/libero_object_no_noops_lerobot")
EMBODIMENT_TAG = "new_embodiment"

device = "cuda" if torch.cuda.is_available() else "cpu"

from gr00t.experiment.data_config import DATA_CONFIG_MAP


data_config = DATA_CONFIG_MAP["franka"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

# print out the policy model architecture
print(policy.model)


import numpy as np

modality_config = policy.modality_config

print(modality_config.keys())

for key, value in modality_config.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
    else:
        print(key, value)


# Create the dataset
dataset = LiberoSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="torchvision_av",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)


import numpy as np

step_data = dataset[7]

print(step_data)

print("\n\n ====================================")
for key, value in step_data.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
    else:
        print(key, value)


import time
start_time = time.time()
predicted_action = policy.get_action(step_data)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"-----------代码段运行时间: {elapsed_time} 秒")

import time
start_time = time.time()
predicted_action = policy.get_action(step_data)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"========== 代码段运行时间: {elapsed_time} 秒")
for key, value in predicted_action.items():
    print(key, value.shape)