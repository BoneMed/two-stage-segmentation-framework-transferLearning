import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.HFENet import build_hfenet
from torch.optim import Adam
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn
from typing import Tuple, Union, List


class nnUNetTrainerHFENet(nnUNetTrainer):
    def __init__(
        self, 
        plans: dict, 
        configuration: str, 
        fold: int, 
        dataset_json: dict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 5e-4

        # 给输出文件夹名加个后缀，避免跟别的消融跑的结果覆盖在一起
        # 改这一行的 '_residual' 就能换成这次跑的是哪个配置
        if self.output_folder_base is not None:
            self.output_folder_base = self.output_folder_base + '_residual'
            self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        arch_kwargs = {
            'n_stages': 6,
            'features_per_stage': [32, 64, 128, 256, 320, 320],
            'conv_op': 'torch.nn.modules.conv.Conv3d',
            'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'strides': arch_init_kwargs['strides'],
            'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
            'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
        }

        # ablation switches: override from the shell, e.g. in a SLURM script:
        #   export HFENET_MSGC_MODE=plain                 # 'gated' (default) or 'plain'
        #   export HFENET_USE_RESIDUAL_PURIFICATION=false  # 'true' (default) or 'false'
        msgc_mode = os.environ.get('HFENET_MSGC_MODE', 'gated')
        use_residual_purification = os.environ.get('HFENET_USE_RESIDUAL_PURIFICATION', 'true').lower() == 'true'

        model = build_hfenet(
            arch_kwargs,
            num_input_channels,
            num_output_channels,
            deep_supervision=enable_deep_supervision,
            msgc_mode=msgc_mode,
            use_residual_purification=use_residual_purification,
            )
        
        return model
                                   

class nnUNetTrainerHFENet_500epochs(nnUNetTrainerHFENet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        

class nnUNetTrainerHFENet_2000epochs(nnUNetTrainerHFENet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
        

    

    
    
