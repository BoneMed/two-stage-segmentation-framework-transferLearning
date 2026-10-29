import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.utilities.network_initialization import InitWeights_He
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.regularization import DropPath
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD


class SpatialNorm(nn.Module):
    # Layer normalization for D, W, H
    def __init__(self, input_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_channels))
        self.beta = nn.Parameter(torch.zeros(input_channels))
        self.input_channels = input_channels
        self.eps = eps

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.input_channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    

class MultiScaleGatedConv(nn.Module):
    r""" Multi-scale gated convolution with hierarchical feature aggregation """
    
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 order: int = 5,
                 s: float = 1,
                 ):
        super().__init__()
        
        # Store configuration
        self._scale_factor = s
        self._depth = order
        
        # Compute channel splits for multi-scale branches
        channel_splits = self._compute_channel_hierarchy(input_channels, order)
        self._channel_splits = channel_splits
        total_branch_channels = sum(channel_splits)
        
        # Convert stride to list format
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        
        # Input projection: expand to gate and content branches
        self._input_proj = self._build_pointwise_conv(
            conv_op, input_channels, input_channels * 2, stride, conv_bias
        )
        
        # Spatial mixing with depthwise separable convolution
        self._spatial_mixer = self._build_depthwise_conv(
            conv_op, total_branch_channels, kernel_size=7, 
            stride=stride, bias=conv_bias
        )
        
        # Recursive feature refinement modules
        self._refinement_layers = self._build_refinement_chain(
            conv_op, channel_splits
        )
        
        # Output projection
        self._output_proj = self._build_pointwise_conv(
            conv_op, input_channels, output_channels, (1,1,1), False
        )
    
    def _compute_channel_hierarchy(self, base_channels, depth):
        """Generate channel numbers for each scale level"""
        hierarchy = [base_channels >> i for i in range(depth)]
        return hierarchy[::-1]  # Reverse for ascending order
    
    def _build_pointwise_conv(self, conv_op, in_ch, out_ch, stride, bias):
        """Construct 1x1x1 convolution"""
        kernel = (1, 1, 1)
        pad = tuple((k - 1) // 2 for k in kernel)
        return conv_op(in_ch, out_ch, kernel, stride, padding=pad, dilation=1, bias=bias)
    
    def _build_depthwise_conv(self, conv_op, channels, kernel_size, stride, bias):
        """Construct depthwise separable convolution"""
        k = (kernel_size, kernel_size, kernel_size)
        pad = tuple((dim - 1) // 2 for dim in k)
        return conv_op(channels, channels, k, stride, padding=pad, 
                       dilation=1, bias=bias, groups=channels)
    
    def _build_refinement_chain(self, conv_op, channel_list):
        """Build cascade of 1x1 convolutions for progressive refinement"""
        layers = nn.ModuleList()
        for idx in range(len(channel_list) - 1):
            layers.append(
                conv_op(channel_list[idx], channel_list[idx + 1], (1, 1, 1))
            )
        return layers

    def forward(self, x):
        # Project input to expanded representation
        projected = self._input_proj(x)
        
        # Split into base and multi-scale branches
        base_ch = self._channel_splits[0]
        branch_ch = sum(self._channel_splits)
        base_features, branch_features = projected.split([base_ch, branch_ch], dim=1)
        
        # Apply spatial mixing and scaling
        mixed_features = self._spatial_mixer(branch_features) * self._scale_factor
        
        # Decompose into scale-specific features
        scale_features = mixed_features.split(self._channel_splits, dim=1)
        
        # Progressive feature aggregation
        aggregated = base_features + scale_features[0]
        for layer_idx, refine_layer in enumerate(self._refinement_layers):
            aggregated = refine_layer(aggregated) + scale_features[layer_idx + 1]
        
        # Final projection
        return self._output_proj(aggregated)


class PlainConvBranch(nn.Module):
    r""" Ablation substitute for MultiScaleGatedConv: keeps the same expand -> spatial-mix ->
    project skeleton (so the parameter/FLOPs budget stays close) but with no multi-scale
    channel split, no recursive refinement, and no gating. Drop-in compatible with the
    `gated_conv_module` slot in HierarchicalFeatureBlock. """

    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 order: int = 5,
                 s: float = 1,
                 ):
        super().__init__()
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        hidden_channels = input_channels * 2

        self._expand = conv_op(input_channels, hidden_channels, (1, 1, 1), stride,
                                padding=0, dilation=1, bias=conv_bias)
        self._spatial_mix = conv_op(hidden_channels, hidden_channels, (7, 7, 7), 1,
                                     padding=3, dilation=1, bias=conv_bias, groups=hidden_channels)
        self._project = conv_op(hidden_channels, output_channels, (1, 1, 1), (1, 1, 1),
                                 padding=0, dilation=1, bias=False)

    def forward(self, x):
        x = self._expand(x)
        x = self._spatial_mix(x)
        return self._project(x)


MSGC_REGISTRY = {
    'gated': MultiScaleGatedConv,
    'plain': PlainConvBranch,
}


class HierarchicalFeatureBlock(nn.Module):
    r""" Hierarchical feature enhancement module with gated convolution """
    
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 transform_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 layer_eps: float = 1e-6,
                 drop_path: int = 0,
                 gated_conv_module = MultiScaleGatedConv,
                 order: int = 3,
                 ):

        super().__init__()
        
        # Store essential dimensions
        self._hidden_dim = transform_channels
        
        # Normalize parameters
        ksize = maybe_convert_scalar_to_list(conv_op, kernel_size)
        st = maybe_convert_scalar_to_list(conv_op, stride)
        norm_kwargs = norm_op_kwargs if norm_op_kwargs is not None else {}
        act_kwargs = nonlin_kwargs if nonlin_kwargs is not None else {}

        # Build gated attention branch
        self._gated_attention = gated_conv_module(
            conv_op, transform_channels, transform_channels, st, conv_bias, order
        )
        
        # Normalization layers for feature conditioning  
        self._pre_gate_norm = SpatialNorm(transform_channels, eps=layer_eps)
        self._post_gate_norm = SpatialNorm(transform_channels, eps=layer_eps)

        # Channel projection: input -> hidden
        self._channel_reduce = ConvDropoutNormReLU(
            conv_op, input_channels, transform_channels, 1, 1, conv_bias
        )

        # Feature transformation path
        self._transform_block = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, transform_channels, output_channels, ksize, st, conv_bias,
                None, None, dropout_op, dropout_op_kwargs, nonlin, act_kwargs
            ),
            ConvDropoutNormReLU(
                conv_op, output_channels, output_channels, ksize, st, conv_bias
            )
        )

        # Stochastic depth for regularization
        self._stochastic_skip = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Activation function
        self._activation = nonlin(**act_kwargs) if nonlin is not None else nn.Identity()

        # Learnable scaling parameter for gated branch
        self._gate_scale = None
        if layer_eps > 0:
            self._gate_scale = nn.Parameter(
                torch.full((transform_channels,), layer_eps), requires_grad=True
            )

    def forward(self, x):
        # Preserve input for residual connection
        identity = x
        
        # Project to hidden dimension
        hidden = self._channel_reduce(x)
        
        # Apply gated attention with learnable scaling
        gate_weight = self._gate_scale.reshape(-1, 1, 1, 1)
        normalized_hidden = self._pre_gate_norm(hidden)
        gated_features = gate_weight * self._gated_attention(normalized_hidden)
        hidden = hidden + self._stochastic_skip(gated_features)
        
        # Transform through convolution path
        normalized_features = self._post_gate_norm(hidden)
        transformed = self._transform_block(normalized_features)
        
        # Residual fusion with stochastic depth
        fused = identity + self._stochastic_skip(transformed)
        
        return self._activation(fused)


class DeconvUpsampler(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
    ):
        super().__init__()

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class PyramidFeatureExtractor(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 pool_type: str = 'conv',
                 drop_path_rate = 0.5,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 msgc_mode: str = 'gated',
                 ):

        super().__init__()
        order_range = [2, 2, 2, 3, 3, 4]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(order_range, int):
            order_range = [order_range] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        assert msgc_mode in MSGC_REGISTRY, f"msgc_mode must be one of {list(MSGC_REGISTRY.keys())}, got {msgc_mode}"
        gated_conv_list = [MSGC_REGISTRY[msgc_mode]] * n_stages

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_stages)]

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        # stem function
        stem_channels = features_per_stage[0]
        self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                      norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        input_channels = stem_channels

        stages = []
        encoder_layers = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1
            stage = StackedResidualBlocks(
                2, conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
            )

            if s >= 2:
                encoder_layers.append(
                    HierarchicalFeatureBlock(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        transform_channels=features_per_stage[s]//2,
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        layer_eps=1e-6,
                        drop_path=dp_rates[s],
                        gated_conv_module=gated_conv_list[s],
                        order=order_range[s],
                    )
                )
            else:
                encoder_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.conn_pos = len(n_blocks_per_stage) - 1 - n_blocks_per_stage[::-1].index(2)
        self.SpatialNorm = SpatialNorm(features_per_stage[self.conn_pos], eps=1e-6)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.msgc_mode = msgc_mode

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            if s == self.conn_pos:
                x = self.SpatialNorm(x)
            x = self.encoder_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class ProgressiveDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 use_residual_purification: bool = True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        self.Init_Residual = BasicBlockD(
            conv_op=encoder.conv_op,
            input_channels=encoder.output_channels[-1],
            output_channels=encoder.output_channels[-1],
            kernel_size=encoder.kernel_sizes[-1],
            stride=1,
            conv_bias=encoder.conv_bias,
            norm_op=encoder.norm_op,
            norm_op_kwargs=encoder.norm_op_kwargs,
            nonlin=encoder.nonlin,
            nonlin_kwargs=encoder.nonlin_kwargs,
        )

        stages = []
        upsample_layers = []
        seg_layers = []
        residual_blocks = []

        # 添加解码器的HierarchicalFeatureBlock层
        decoder_feature_layers = []

        # 计算drop_path_rate（与编码器保持一致）
        drop_path_rate = 0.5
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_stages_encoder)]

        # 计算order_range（与编码器保持一致）
        order_range = list(range(1, n_stages_encoder + 1))

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            encoder_stage_idx = n_stages_encoder - s - 1

            upsample_layers.append(DeconvUpsampler(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))

            if use_residual_purification:
                residual_blocks.append(
                    BasicBlockD(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    )
                )
            else:
                residual_blocks.append(nn.Identity())

            stages.append(
                ResidualUnit(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                )
            )

            if encoder_stage_idx >= 2:
                decoder_feature_layers.append(
                    HierarchicalFeatureBlock(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        transform_channels=input_features_skip // 2,
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                        layer_eps=1e-6,
                        drop_path=dp_rates[encoder_stage_idx],
                        gated_conv_module=MSGC_REGISTRY[encoder.msgc_mode],
                        order=order_range[encoder_stage_idx],
                    )
                )
            else:
                decoder_feature_layers.append(nn.Identity())

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

        self.decoder_feature_layers = nn.ModuleList(decoder_feature_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        lres_input = self.Init_Residual(lres_input)
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            res = self.residual_blocks[s](skips[-(s + 2)])
            x = torch.cat((x, res), 1)
            x = self.stages[s](x)

            x = self.decoder_feature_layers[s](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output

    
class HFENet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 msgc_mode: str = 'gated',
                 use_residual_purification: bool = True,
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PyramidFeatureExtractor(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            msgc_mode=msgc_mode,
        )

        self.decoder = ProgressiveDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                           use_residual_purification=use_residual_purification)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


def build_hfenet(
        arch_kwargs,
        input_channels,
        output_channels,
        deep_supervision: Union[bool, None] = None,
        msgc_mode: str = 'gated',
        use_residual_purification: bool = True,
):
            
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """

    num_stages = arch_kwargs['n_stages']

    dim = len(arch_kwargs['kernel_sizes'][0])
    conv_op = convert_dim_to_conv_op(dim)

    segmentation_network_class_name = 'HFENet'
    network_class = HFENet
    kwargs = {
        'HFENet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': arch_kwargs['n_conv_per_stage'],
        'n_conv_per_stage_decoder': arch_kwargs['n_conv_per_stage_decoder']
    }

    model = network_class(
        input_channels=input_channels,
        n_stages=num_stages,
        features_per_stage = arch_kwargs['features_per_stage'],
        conv_op=conv_op,
        kernel_sizes=arch_kwargs['kernel_sizes'],
        strides=arch_kwargs['strides'],
        num_classes=output_channels,
        deep_supervision=deep_supervision,
        msgc_mode=msgc_mode,
        use_residual_purification=use_residual_purification,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    
    return model

