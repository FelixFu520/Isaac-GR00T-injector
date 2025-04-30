from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LiberoSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP

dataset_path = "/root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/data/IPEC-COMMUNITY/libero_object_no_noops_lerobot"   # change this to your dataset path

data_config = DATA_CONFIG_MAP["franka"]

dataset = LiberoSingleDataset(
    dataset_path=dataset_path,
    modality_configs=data_config.modality_config(),
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
)

resp = dataset[7]
any_describe(resp)