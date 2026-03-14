from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import pdb


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            #Union表示既可以接受nn.Module类型，也可以接受Dict[str,nn.Module]类型
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            graph_model: nn.Module = None,  # 新增graph_model参数
            pointcloud_film_model: nn.Module = None,  # 点云FiLM调制模型
            tactile_model: nn.Module = None,  # 触觉encoder
            amam_model: nn.Module = None,  # AMAM多模态融合模块
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            #比如现在我既有head_cam, 也有front_cam...图像
            #使用共享rgb模型的意思就是把head_cam与front_cam的图像先融合后，统一输入到一个resnet网络做处理
            #非共享的意思就是，分别用一个单独的resnet或其他网络处理head_cam, front_cam图像，最后再合并特征
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False
        ):
        """
        Assumes rgb input: B,C,H,W (192,3,240,320)
        Assumes low_dim input: B,D (192,8)
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        tcp_keys = list()  # 新增tcp_keys列表
        pointcloud_keys = list()  # 点云keys
        tactile_keys = list()  # 触觉keys
        #示例: m = nn.ModuleDict()
        # m['head_cam'] = resnet18
        # m['left_cam'] = resnet34
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            #形状是列表这里转化成元组
            shape = tuple(attr['shape'])
            #如果有type则读，没有就默认为low_dim
            # 2.判断是 RGB 图 (type='rgb') 还是低维量 (type='low_dim')
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                # 3.先把这个键名记到 rgb_keys
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                #如果不share模型，就需要为每个视觉图像key用单独的模型
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        # 如果用户传入了一个字典，就取 rgb_model[key]
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        # 否则用户只给了一个 nn.Module，给这个通道 deepcopy 一份
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    #归一化
                    if use_group_norm:
                        #把模型替换到group norm
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    #把对应的rgb模型存到key_model_map中
                    key_model_map[key] = this_model
                
                # configure resize
                #做resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                #key_transform_map是做图像预处理的
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == 'tcp':  # 新增TCP类型处理
                tcp_keys.append(key)
                # TCP数据不需要添加到key_model_map，使用共享的graph_model
            elif type == 'pointcloud':  # 点云类型
                pointcloud_keys.append(key)
            elif type == 'tactile':  # 触觉类型
                tactile_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        tcp_keys = sorted(tcp_keys)  # 排序tcp_keys
        pointcloud_keys = sorted(pointcloud_keys)
        tactile_keys = sorted(tactile_keys)

        self.shape_meta = shape_meta
        #为head_cam, front_cam..(如有)储存对应的模型
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.tcp_keys = tcp_keys  # 存储tcp_keys
        self.pointcloud_keys = pointcloud_keys  # 存储点云keys
        self.tactile_keys = tactile_keys  # 存储触觉keys
        self.key_shape_map = key_shape_map

        # 存储共享的graph_model（只需要一个实例）
        self.graph_model = graph_model
        # 存储点云FiLM调制模型
        self.pointcloud_film_model = pointcloud_film_model
        # 存储触觉encoder（所有agent共享同一个encoder权重）
        self.tactile_model = tactile_model
        # 存储AMAM多模态融合模块
        self.amam_model = amam_model

    def forward(self, obs_dict):
        batch_size = None
        features = list()

        # ===== 1. Extract vision features (ResNet + FiLM) =====
        vision_feature = None
        if self.share_rgb_model:
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map['rgb'](imgs)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            feature = torch.moveaxis(feature,0,1)
            vision_feature = feature.reshape(batch_size,-1)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)

                # 点云FiLM调制: 用点云特征增强图像特征
                if self.pointcloud_film_model is not None and len(self.pointcloud_keys) > 0:
                    pc_key = self.pointcloud_keys[0]
                    point_cloud = obs_dict[pc_key]
                    feature = self.pointcloud_film_model(point_cloud, feature)

                vision_feature = feature  # (B, 512)

        # ===== 2. Extract graph features (GNN) =====
        graph_feature = None
        if len(self.tcp_keys) > 0:
            tcp_data_list = []
            for key in self.tcp_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                assert data.shape[1:] == self.key_shape_map[key]
                tcp_data_list.append(data)
            tcp_combined = torch.stack(tcp_data_list, dim=1)
            if self.graph_model is not None:
                graph_feature = self.graph_model(tcp_combined)  # (B, 64)

        # ===== 3. Extract tactile features =====
        tactile_feature = None
        if self.tactile_model is not None and len(self.tactile_keys) > 0:
            tactile_features_list = []
            for key in self.tactile_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                tactile_features_list.append(self.tactile_model(data))
            # 通常只有1个tactile key, 如果多个则拼接
            tactile_feature = torch.cat(tactile_features_list, dim=-1)  # (B, 64)

        # ===== 4. AMAM fusion or direct concatenation =====
        if self.amam_model is not None and vision_feature is not None \
                and tactile_feature is not None and graph_feature is not None:
            # AMAM: 动态加权三个模态后拼接
            fused = self.amam_model(vision_feature, tactile_feature, graph_feature)
            features.append(fused)
        else:
            # 无AMAM时直接拼接（向后兼容）
            if vision_feature is not None:
                features.append(vision_feature)
            if graph_feature is not None:
                features.append(graph_feature)
            if tactile_feature is not None:
                features.append(tactile_feature)

        # ===== 5. Low-dim features (agent_pos) — 不参与AMAM =====
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        # AMAM: (B, 512+64+64) + (B, 8) = (B, 648)
        # No AMAM: (B, 512) + (B, 8) + (B, 64) + (B, 64) = (B, 648)
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                #元组操作: (1,) + (3, 224, 224) -> (1, 3, 224, 224)
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        #返回520+64=584 (如果GNN output_dim=64)
        output_shape = example_output.shape[1:]
        return output_shape
