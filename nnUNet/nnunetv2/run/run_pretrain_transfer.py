"""Transfer learning utilities for pre-trained model adaptation."""

from typing import Dict, Any, List
from unittest.mock import patch
import torch
from torch.nn import Module
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

from run_training import run_training_entry


class PretrainedWeightAdapter:
    """Handles loading and adapting pretrained weights to target network."""
    
    EXCLUDED_PATTERNS: List[str] = ['seg_layers']
    
    STEM_WEIGHT_KEYS: List[str] = [
        'encoder.stem.convs.0.conv.weight',
        'encoder.stem.convs.0.all_modules.0.weight',
        'decoder.encoder.stem.convs.0.conv.weight',
        'decoder.encoder.stem.convs.0.all_modules.0.weight',
    ]
    
    def __init__(self, checkpoint_path: str, verbose: bool = False):
        self._ckpt_path = checkpoint_path
        self._verbose = verbose
        self._pretrained_weights = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, torch.Tensor]:
        """Load weights from checkpoint file."""
        ckpt_data = torch.load(self._ckpt_path, weights_only=False)
        
        if self._ckpt_path.endswith('pth'):
            return ckpt_data['network_weights']
        elif self._ckpt_path.endswith('model'):
            return ckpt_data['state_dict']
        
        raise ValueError(f"Unsupported checkpoint format: {self._ckpt_path}")
    
    @staticmethod
    def _unwrap_model(model: Module) -> Module:
        """Extract base model from DDP or compiled wrappers."""
        unwrapped = model
        
        if isinstance(unwrapped, DistributedDataParallel):
            unwrapped = unwrapped.module
        
        if isinstance(unwrapped, OptimizedModule):
            unwrapped = unwrapped._orig_mod
        
        return unwrapped
    
    def _adapt_input_channels(self, target_state: Dict[str, torch.Tensor]) -> None:
        """Adjust pretrained weights for different input channel counts."""
        ref_key = 'encoder.stem.convs.0.conv.weight'
        
        n_target_ch = target_state[ref_key].shape[1]
        n_pretrain_ch = self._pretrained_weights[ref_key].shape[1]
        
        if n_pretrain_ch <= n_target_ch:
            return
        
        # Truncate pretrained weights to match target channels
        for weight_key in self.STEM_WEIGHT_KEYS:
            if weight_key in self._pretrained_weights:
                original = self._pretrained_weights[weight_key]
                self._pretrained_weights[weight_key] = original[:, :n_target_ch, :, :, :]
    
    def _should_skip_key(self, param_name: str) -> bool:
        """Check if parameter should be excluded from transfer."""
        return any(pattern in param_name for pattern in self.EXCLUDED_PATTERNS)
    
    def _validate_compatibility(self, target_state: Dict[str, torch.Tensor]) -> None:
        """Ensure pretrained weights are compatible with target architecture."""
        for param_name, param_tensor in target_state.items():
            if self._should_skip_key(param_name):
                continue
            
            if param_name not in self._pretrained_weights:
                raise AssertionError(
                    f"Key {param_name} is missing in the pretrained model weights. "
                    f"The pretrained weights do not seem to be compatible with your network."
                )
            
            pretrain_shape = self._pretrained_weights[param_name].shape
            target_shape = param_tensor.shape
            
            if pretrain_shape != target_shape:
                raise AssertionError(
                    f"The shape of the parameters of key {param_name} is not the same. "
                    f"Pretrained model: {pretrain_shape}; your network: {target_shape}. "
                    f"The pretrained model does not seem to be compatible with your network."
                )
    
    def _filter_transferable_weights(
        self, 
        target_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get subset of pretrained weights that can be transferred."""
        transferable = {}
        
        for key, weight in self._pretrained_weights.items():
            is_in_target = key in target_state
            is_allowed = not self._should_skip_key(key)
            
            if is_in_target and is_allowed:
                transferable[key] = weight
        
        return transferable
    
    def _log_transferred_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Print information about transferred weights."""
        if not self._verbose:
            return
        
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in weights.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    
    def transfer_to(self, network: Module) -> None:
        """Transfer pretrained weights to target network."""
        base_model = self._unwrap_model(network)
        current_state = base_model.state_dict()
        
        # Adapt for channel mismatch
        self._adapt_input_channels(current_state)
        
        # Verify compatibility
        self._validate_compatibility(current_state)
        
        # Filter and transfer weights
        weights_to_transfer = self._filter_transferable_weights(current_state)
        
        print(f"################### Loading pretrained weights from file {self._ckpt_path} ###################")
        self._log_transferred_weights(weights_to_transfer)
        
        # Apply transferred weights
        current_state.update(weights_to_transfer)
        base_model.load_state_dict(current_state)


def initialize_from_pretrained(network: Module, ckpt_path: str, verbose: bool = False) -> None:
    """Entry point for loading pretrained weights into network."""
    adapter = PretrainedWeightAdapter(ckpt_path, verbose)
    adapter.transfer_to(network)


if __name__ == '__main__':
    with patch("run_training.load_pretrained_weights", initialize_from_pretrained):
        run_training_entry()