import torch
from torch import nn
from typing import Tuple, Union, List

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunet_mednext import create_mednext_v1


class nnUNetTrainerMedNeXt(nnUNetTrainer):
    """MedNeXt-L (kernel 3, official MICCAI 2023 config) inside the nnU-Net v2 pipeline.

    Optimiser follows the official MedNeXt trainer (AdamW, lr 1e-3, eps 1e-4) with
    nnU-Net's poly LR decay; data, preprocessing and splits are nnU-Net's own.
    """

    mednext_model_id = 'L'
    mednext_kernel_size = 3
    mednext_grad_checkpointing = True

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # Always build the deep-supervision heads so checkpoints load the same way at train and
        # inference time (nnU-Net's predictor builds with enable_deep_supervision=False);
        # do_ds then decides whether the extra outputs are returned.
        model = create_mednext_v1(
            num_input_channels,
            num_output_channels,
            nnUNetTrainerMedNeXt.mednext_model_id,
            kernel_size=nnUNetTrainerMedNeXt.mednext_kernel_size,
            deep_supervision=True,
        )
        model.do_ds = enable_deep_supervision
        # The official L config enables gradient checkpointing purely to save memory; turning it
        # off gives identical results and is faster on large-memory GPUs (A100 80G / H100).
        model.outside_block_checkpointing = nnUNetTrainerMedNeXt.mednext_grad_checkpointing
        return model

    def _get_deep_supervision_scales(self):
        # MedNeXt always downsamples isotropically 4 times and emits 5 outputs,
        # independent of the nnU-Net plans' pooling strides.
        if self.enable_deep_supervision:
            return [[1.0, 1.0, 1.0], [0.5] * 3, [0.25] * 3, [0.125] * 3, [0.0625] * 3]
        return None

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, torch._dynamo.OptimizedModule):
            mod = mod._orig_mod
        mod.do_ds = enabled

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr,
                                      weight_decay=self.weight_decay, eps=1e-4)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerMedNeXt_2000epochs(nnUNetTrainerMedNeXt):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
