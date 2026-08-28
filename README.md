

# Isaac-GR00T-injector
Isaac-GR00T补充和学习
笔记链接: https://siwrc302o4r.feishu.cn/wiki/C4q0wUThZijumVkVWLfctAmAnxf?fromScene=spaceOverview

## 跑通getting_started
- 加载数据，0_load_dataset.ipynb
- 加载数据，0_load_dataset.py
- 推理，1_gr00t_inference.ipynb
- 推理，1_gr00t_inference.py
- 微调，2_finetuning.ipynb
- 微调，3_new_embodiment_finetuning.ipynb
- 数据格式，4_deeper_understanding.md
- 部署，5_policy_deployment.md

## 使用libero数据微调并仿真
### 修改代码
把下面代码放到/root/data/projects/embodiedai/grootn1/Isaac-GR00T/gr00t/experiment/data_config.py中
```
class FrankaDataConfig(BaseDataConfig):
    video_keys = ["video.image", "video.wrist_image"]
    state_keys = ["state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw", "state.gripper"]
    action_keys = ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class So100BoxDataConfig(BaseDataConfig):
    video_keys = ["video.laptop", "video.phone"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

DATA_CONFIG_MAP = {
    "gr1_arms_waist": Gr1ArmsWaistDataConfig(),
    "gr1_arms_only": Gr1ArmsOnlyDataConfig(),
    "gr1_full_upper_body": Gr1FullUpperBodyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "single_panda_gripper": SinglePandaGripperDataConfig(),
    "so100": So100DataConfig(),
    "franka": FrankaDataConfig(),
    "so100_box": So100BoxDataConfig(),
}
```

把下面代码放到/root/data/projects/embodiedai/grootn1/Isaac-GR00T/gr00t/data/dataset.py中
```

class LiberoSingleDataset(LeRobotSingleDataset):
    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("rgb")]
                fps = le_video_meta["info"]["video.fps"]
            except ValueError:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            le_statistics = calculate_dataset_statistics(parquet_files)
            with open(stats_path, "w") as f:
                json.dump(le_statistics, f, indent=4)
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata
```

### 微调
- 查看数据，6_franka_dataset.py
- 查看数据&微调， 6_franka_finetuning.ipynb
- 推理，6_franka_inference.py

### 仿真
说明：libero是一个继承了robosuite(MUJOCO)仿真环境的benchmark， 我们可以让机器人的policy在这个仿真环境下运行，来测试模型的效果。
也就是说使用上一步微调出来的模型启动一个web服务， 然后客户端可以向这个服务发送状态、图片，然后从这个服务中获取action， 
客户端获得action后可以在libero中，将这个action渲染成仿真环境中，最终会得到成功率和视频。


因此， 需要安装两个环境，一个是服务端环境，一个是客户端环境
#### 环境安装
**服务端环境安装**
服务端使用的是训练环境也就是gr00t环境， 然后需要在里面额外安装个openpi的openpi-client https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client， 已经下载到这个项目中了
```
cd /root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/server/openpi-client
conda activate gr00t
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

```

**客户端环境安装**
客户端有3部分，
第1个，openpi官方的启动程序https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero,

第2个， libero库https://github.com/Lifelong-Robot-Learning/LIBERO，

第3个， openpi的openpi-client，和服务端一样

下载后组织成sim文件夹的格式，然后安装libero环境
```
conda create -n gr00t_sim python=3.8
conda activate gr00t_sim

# 安装libero
cd /root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/sim/third_party/libero
apt update
apt install cmake
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple robosuite==1.4.1 tyro

# 安装openpi-client
cd /root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/sim/openpi-client
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

```

#### 启动服务
启动服务端服务
```
python /root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/server/serve_policy.py --server --model_path /root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/output/6_franka_libero_object_no_noops_lerobot_20000/checkpoint-5000 --embodiment_tag new_embodiment --data_config franka --denoising_steps 4
```

启动客户端程序进行仿真
```
cd /root/data/projects/embodiedai/grootn1/Isaac-GR00T/Isaac-GR00T-injector/sim/libero
python  main.py
```

## 使用自己采集的数据微调，并在so100上部署

## 在grootn1上增加模块
