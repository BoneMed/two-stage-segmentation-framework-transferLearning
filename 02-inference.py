"""
两阶段脊椎分割推理脚本
Stage 1: 低分辨率模型进行裁剪定位
Stage 2: 高分辨率模型进行精准分割
"""

import os

# 必须在 import nnUNetPredictor / 构建网络之前设置：build_network_architecture 是实时读这两个
# 环境变量来决定建哪种架构的。这里写死成 Residual Purification Block 消融（3D-MSGC 保留，
# skip connection 上的 residual purification block 关掉）对应的配置，跟 Dataset602
# ..._residual 这个 checkpoint 训练时一致。
os.environ['HFENET_MSGC_MODE'] = 'gated'
os.environ['HFENET_USE_RESIDUAL_PURIFICATION'] = 'true'

import sys
import torch
import shutil
import tempfile
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# ==================== 配置 ====================
class Config:
    # 环境变量
    BASE_PATH = Path("/mnt/PhD/pythonproject/PHD_PROGRAM/project0121/Data")

    # Stage 1: 低分辨率裁剪模型（跟其它消融变体共用同一个，不受 MSGC 消融影响）
    STAGE1_MODEL = {
        "path": str(BASE_PATH / 'nnUNet_results/Dataset101_verseg/nnUNetTrainer__nnUNetPlans__3d_fullres_my_spacing_Baseline'),
        "folds": ('0',),
        "checkpoint": "checkpoint_best.pth",
        "spacing": [6.0, 6.0, 6.0],  # 低分辨率spacing
    }

    # Stage 2: 高分辨率分割模型 —— Residual Purification Block 消融版本
    STAGE2_MODEL = {
        "path": str(BASE_PATH / 'nnUNet_results/Dataset602_verseg_cropped/nnUNetTrainerMedNeXt_2000epochs__nnUNetPlans__3d_fullres'),
        "folds": ('0', '1', '2', '3', '4'),
        "checkpoint": "checkpoint_final.pth",
    }

    # 裁剪参数
    CROP_MARGINS = [30, 30, 30]  # X, Y, Z 方向的margin (mm)

    # 输入输出路径：在 Dataset601_verseg 自己的测试集上跑，跟已有的
    # labelsTs_pred_FENet_trans_0.005 等几个两阶段结果放在同一级，方便直接对比
    INPUT_DIR = str(BASE_PATH / 'nnUNet_raw/Dataset701_gulou/imagesTs')
    OUTPUT_DIR = str(BASE_PATH / 'nnUNet_raw/Dataset701_gulou/labelsTs_pred_MedNeXt')
    TEMP_DIR = './temp_inference_701_labelsTs_pred_MedNeXt'

    # 后处理参数
    MIN_LABEL_VOLUME_CM3 = 5.0  # 最小标签体积阈值 (cm³)


# ==================== 工具函数 ====================
def resample_image(image, target_spacing, is_label=False):
    """重采样图像到目标spacing"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    
    return resampler.Execute(image)


def compute_crop_bounds(mask, image, margins):
    """
    根据mask计算裁剪边界
    
    Parameters:
        mask: 低分辨率分割mask
        image: 原始高分辨率图像
        margins: [margin_x, margin_y, margin_z] in mm
    
    Returns:
        crop_coords: 裁剪坐标字典
    """
    mask_arr = sitk.GetArrayFromImage(mask)
    nonzero = np.nonzero(mask_arr)
    
    if len(nonzero[0]) == 0:
        raise ValueError("Empty mask, no foreground detected!")
    
    # numpy数组顺序: (z, y, x)
    z_min, z_max = int(nonzero[0].min()), int(nonzero[0].max())
    y_min, y_max = int(nonzero[1].min()), int(nonzero[1].max())
    x_min, x_max = int(nonzero[2].min()), int(nonzero[2].max())
    
    # 转换到物理坐标
    pt_min = mask.TransformIndexToPhysicalPoint((x_min, y_min, z_min))
    pt_max = mask.TransformIndexToPhysicalPoint((x_max, y_max, z_max))
    
    # 映射到原始图像索引
    idx_min = image.TransformPhysicalPointToIndex(pt_min)
    idx_max = image.TransformPhysicalPointToIndex(pt_max)
    
    # 计算margin (像素)
    spacing = image.GetSpacing()
    size = image.GetSize()
    
    margin_px = [int(margins[i] / spacing[i]) for i in range(3)]
    
    # 计算裁剪范围 (带margin和边界检查)
    crop_coords = {
        'x_start': max(0, idx_min[0] - margin_px[0]),
        'x_end': min(size[0], idx_max[0] + margin_px[0]),
        'y_start': max(0, idx_min[1] - margin_px[1]),
        'y_end': min(size[1], idx_max[1] + margin_px[1]),
        'z_start': max(0, idx_min[2] - margin_px[2]),
        'z_end': min(size[2], idx_max[2] + margin_px[2]),
    }
    
    return crop_coords


def crop_image(image, crop_coords):
    """执行图像裁剪"""
    return image[
        crop_coords['x_start']:crop_coords['x_end'],
        crop_coords['y_start']:crop_coords['y_end'],
        crop_coords['z_start']:crop_coords['z_end']
    ]


def restore_to_original_size(cropped_pred, original_image, crop_coords):
    """将裁剪后的预测结果恢复到原始尺寸"""
    # 创建全零背景
    full_arr = np.zeros(
        sitk.GetArrayFromImage(original_image).shape, 
        dtype=np.uint8
    )
    
    # 获取裁剪预测的数组
    cropped_arr = sitk.GetArrayFromImage(cropped_pred)
    
    # 填充到对应位置 (注意numpy数组顺序是z,y,x)
    full_arr[
        crop_coords['z_start']:crop_coords['z_end'],
        crop_coords['y_start']:crop_coords['y_end'],
        crop_coords['x_start']:crop_coords['x_end']
    ] = cropped_arr
    
    # 转换回SimpleITK格式
    full_pred = sitk.GetImageFromArray(full_arr)
    full_pred.CopyInformation(original_image)
    
    return full_pred


def remove_small_labels(image, min_volume_cm3=5.0):
        """
        后处理：移除体积过小的标签
        
        Parameters:
            image: SimpleITK图像 (分割mask)
            min_volume_cm3: 最小体积阈值 (cm³)
        
        Returns:
            处理后的SimpleITK图像
        """
        data = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        
        # 计算单个体素体积 (cm³)
        voxel_volume_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        
        # 计算最小体素数阈值
        min_voxels = int(min_volume_cm3 / voxel_volume_cm3)
        
        labels = np.unique(data)
        labels = labels[labels > 0]
        
        removed_labels = []
        for label in labels:
            voxel_count = np.sum(data == label)
            volume_cm3 = voxel_count * voxel_volume_cm3
            
            if volume_cm3 < min_volume_cm3:
                data[data == label] = 0
                removed_labels.append((int(label), f"{volume_cm3:.2f}"))
        
        if removed_labels:
            print(f"   🧹 移除小标签: {removed_labels}")
        
        # 创建新图像
        result = sitk.GetImageFromArray(data)
        result.CopyInformation(image)
        
        return result
        
# ==================== 推理器类 ====================
class TwoStagePredictor:
    """两阶段脊椎分割推理器"""
    
    def __init__(self, config):
        self.config = config
        self.stage1_predictor = None
        self.stage2_predictor = None
        
        # 创建临时目录
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def _init_predictor(self, model_config, name):
        """初始化单个nnUNet预测器"""
        print(f"🔧 初始化 {name} 模型...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        predictor.initialize_from_trained_model_folder(
            model_config["path"],
            use_folds=model_config["folds"],
            checkpoint_name=model_config["checkpoint"],
        )
        
        print(f"✅ {name} 模型加载完成!")
        return predictor
    
    def initialize(self):
        """初始化两个阶段的模型"""
        self.stage1_predictor = self._init_predictor(
            self.config.STAGE1_MODEL, "Stage 1 (裁剪定位)"
        )
        self.stage2_predictor = self._init_predictor(
            self.config.STAGE2_MODEL, "Stage 2 (精准分割)"
        )
        print("🎉 两阶段模型初始化完成!\n")
    
    def _run_inference(self, predictor, input_path, output_dir, case_name):
        """运行单次nnUNet推理"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{case_name}.nii.gz")
        
        predictor.predict_from_files(
            [[input_path]],
            [output_path],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )
        
        return output_path
    
    def predict_single(self, input_path, case_name):
        """
        对单个病例进行两阶段预测
        
        Returns:
            output_path: 最终预测结果路径
        """
        print(f"\n{'='*60}")
        print(f"🔬 处理: {case_name}")
        
        temp_dir = self.config.TEMP_DIR
        
        # ========== Stage 1: 低分辨率分割用于裁剪 ==========
        print("\n📍 Stage 1: 低分辨率分割...")
        
        # 读取原始图像
        original_image = sitk.ReadImage(input_path)
        print(f"   原始图像: size={original_image.GetSize()}, spacing={[f'{s:.2f}' for s in original_image.GetSpacing()]}")
        
        # 重采样到低分辨率
        low_res_spacing = self.config.STAGE1_MODEL["spacing"]
        low_res_image = resample_image(original_image, low_res_spacing)
        print(f"   低分辨率图像: size={low_res_image.GetSize()}")
        
        # 保存临时文件
        temp_low_res_path = os.path.join(temp_dir, f"{case_name}_0000.nii.gz")
        sitk.WriteImage(low_res_image, temp_low_res_path)
        
        # 运行Stage 1推理
        stage1_output = self._run_inference(
            self.stage1_predictor, temp_low_res_path, temp_dir, case_name
        )
        
        # 读取分割结果
        spine_mask = sitk.ReadImage(stage1_output)
        mask_arr = sitk.GetArrayFromImage(spine_mask)
        
        # 二值化
        mask_arr[mask_arr > 0] = 1
        spine_mask_binary = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        spine_mask_binary.CopyInformation(spine_mask)
        
        print(f"   Stage 1 分割完成, 前景体素: {np.sum(mask_arr > 0)}")
        
        # 检查是否找到脊椎
        if np.sum(mask_arr) == 0:
            print(f"   ⚠️ 未检测到脊椎区域!")
            return None
        
        # ========== 计算裁剪坐标 ==========
        print("\n✂️ 计算裁剪区域...")
        
        try:
            crop_coords = compute_crop_bounds(
                spine_mask_binary, 
                original_image, 
                self.config.CROP_MARGINS
            )
            print(f"   裁剪坐标: X=[{crop_coords['x_start']}:{crop_coords['x_end']}], "
                  f"Y=[{crop_coords['y_start']}:{crop_coords['y_end']}], "
                  f"Z=[{crop_coords['z_start']}:{crop_coords['z_end']}]")
        except Exception as e:
            print(f"   ❌ 裁剪失败: {e}")
            return None
        
        # 执行裁剪
        cropped_image = crop_image(original_image, crop_coords)
        print(f"   裁剪后图像: size={cropped_image.GetSize()}")
        
        # ========== Stage 2: 高分辨率精准分割 ==========
        print("\n🎯 Stage 2: 高分辨率精准分割...")
        
        # 保存裁剪后的图像
        temp_cropped_path = os.path.join(temp_dir, f"{case_name}_cropped_0000.nii.gz")
        sitk.WriteImage(cropped_image, temp_cropped_path)
        
        # 运行Stage 2推理
        stage2_output = self._run_inference(
            self.stage2_predictor, temp_cropped_path, temp_dir, f"{case_name}_cropped"
        )
        
        # 读取裁剪区域的预测结果
        cropped_prediction = sitk.ReadImage(stage2_output)
        print(f"   Stage 2 分割完成")
        
        # ========== 恢复到原始尺寸 ==========
        print("\n🔄 恢复到原始尺寸...")
        
        full_prediction = restore_to_original_size(
            cropped_prediction, original_image, crop_coords
        )

        # ========== 后处理：移除小标签 ==========
        print("\n🧹 后处理: 过滤小标签...")
        full_prediction = remove_small_labels(
            full_prediction, 
            min_volume_cm3=self.config.MIN_LABEL_VOLUME_CM3
        )
        
        # 保存最终结果
        output_path = os.path.join(self.config.OUTPUT_DIR, f"{case_name}.nii.gz")
        sitk.WriteImage(full_prediction, output_path)
        
        print(f"   ✅ 保存: {output_path}")
        print(f"   最终预测: size={full_prediction.GetSize()}")
        
        # 清理临时文件
        for f in [temp_low_res_path, stage1_output, temp_cropped_path, stage2_output]:
            if os.path.exists(f):
                os.remove(f)
        
        return output_path
    
    def predict_batch(self, input_dir):
        """批量预测"""
        input_path = Path(input_dir)
        image_files = sorted(input_path.glob("*.nii.gz"))
        
        print(f"📊 发现 {len(image_files)} 个图像文件")
        
        results = {"success": [], "failed": []}
        
        for i, img_file in enumerate(image_files, 1):
            case_name = img_file.stem.replace("_0000", "").replace(".nii", "")
            print(f"\n[{i}/{len(image_files)}] ", end="")
            
            try:
                output = self.predict_single(str(img_file), case_name)
                if output:
                    results["success"].append(case_name)
                else:
                    results["failed"].append(case_name)
            except Exception as e:
                print(f"❌ 错误: {e}")
                results["failed"].append(case_name)
        
        # 汇总
        print("\n" + "=" * 60)
        print("🎉 批量预测完成!")
        print(f"   ✅ 成功: {len(results['success'])}")
        print(f"   ❌ 失败: {len(results['failed'])}")
        
        if results["failed"]:
            print(f"   失败案例: {results['failed']}")
        
        return results

    # ==================== 在工具函数部分添加 ====================



# ==================== 主函数 ====================
def main():
    print("🚀 两阶段脊椎分割推理")
    print("   Stage 1: 低分辨率定位裁剪")
    print("   Stage 2: 高分辨率精准分割\n")
    
    # 设置环境变量
    os.environ['nnUNet_raw'] = str(Config.BASE_PATH / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(Config.BASE_PATH / 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(Config.BASE_PATH / 'nnUNet_results')
    
    # 检查CUDA
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🚀 CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}\n")
    
    # 初始化预测器
    predictor = TwoStagePredictor(Config)
    predictor.initialize()
    
    # 批量预测
    predictor.predict_batch(Config.INPUT_DIR)


if __name__ == "__main__":
    main()