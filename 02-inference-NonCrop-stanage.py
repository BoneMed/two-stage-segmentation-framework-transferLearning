"""
单阶段脊椎分割推理脚本（对照用，无裁剪）—— Stanage 版
直接用在未裁剪全图（Dataset601_verseg）上训练的模型做分割，不经过 Stage 1 定位裁剪。
跟 02-inference.py（两阶段）用同一批输入数据跑，方便对比 Dice/HD95/RVD 和推理耗时。

本机内存不够跑这个（超大全图 case 合并 logits 能吃四五十 GB），所以搬到 Stanage 跑。
运行前 cd 到 /mnt/parscratch/users/cov23yc/project0121 下再执行：
    python 02-inference-NonCrop.py
"""

import os

# 必须在 import nnUNetPredictor / 构建网络之前设置：build_network_architecture 是实时读这两个
# 环境变量来决定建哪种架构的（训练时的配置不会存在 checkpoint/plans.json 里）。这里写死成
# Dataset601_verseg 这个 one-stage baseline 实际训练时用的配置（完整 HFENet，没有做消融），
# 不依赖外部终端是否残留了别的消融实验设的 HFENET_* 环境变量。
os.environ['HFENET_MSGC_MODE'] = 'gated'
os.environ['HFENET_USE_RESIDUAL_PURIFICATION'] = 'true'

import time
from pathlib import Path
import torch
import SimpleITK as sitk
import numpy as np
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# ==================== 配置 ====================
class Config:
    # 环境变量
    BASE_PATH = Path("/mnt/parscratch/users/cov23yc/project0121/Data")

    # 单阶段模型：直接在未裁剪全图上训练的 HFENet
    MODEL = {
        "path": str(BASE_PATH / 'nnUNet_results/Dataset601_verseg/nnUNetTrainerHFENet_2000epochs__nnUNetPlans__3d_fullres'),
        "folds": ('0', '1', '2', '3', '4'),
        "checkpoint": "checkpoint_best.pth",
    }

    # 输入输出路径：在 Dataset601_verseg 自己的测试集上跑（未裁剪全图），预测结果跟两阶段
    # 那几个 labelsTs_pred_FENet_trans_* 文件夹放在同一级，方便直接对比
    # 注意：Stanage 这边真正的 40 个测试 case 放在 imagesTsori 下，不是 imagesTs
    # （imagesTs 在 Stanage 上是另外两个不相关的文件，跟本地不是一回事，别搞混了）
    INPUT_DIR = str(BASE_PATH / 'nnUNet_raw/Dataset601_verseg/imagesTsori')
    OUTPUT_DIR = str(BASE_PATH / 'nnUNet_raw/Dataset601_verseg/labelsTs_pred_FENet_without_crop')
    TEMP_DIR = './temp_inference_singlestage'

    # 后处理参数
    MIN_LABEL_VOLUME_CM3 = 5.0  # 最小标签体积阈值 (cm³)


# ==================== 工具函数 ====================
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
class SingleStagePredictor:
    """单阶段脊椎分割推理器（无裁剪，直接在全图上分割）"""

    def __init__(self, config):
        self.config = config
        self.predictor = None

        os.makedirs(config.TEMP_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        self.timings = []  # 记录每个 case 的推理耗时，用于效率对比

    def initialize(self):
        """初始化模型"""
        print("🔧 初始化单阶段模型...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        self.predictor.initialize_from_trained_model_folder(
            self.config.MODEL["path"],
            use_folds=self.config.MODEL["folds"],
            checkpoint_name=self.config.MODEL["checkpoint"],
        )

        print("✅ 单阶段模型加载完成!\n")

    def predict_single(self, input_path, case_name):
        """
        对单个病例直接做单阶段分割（无裁剪）

        Returns:
            output_path: 最终预测结果路径
        """
        print(f"\n{'='*60}")
        print(f"🔬 处理: {case_name}")

        original_image = sitk.ReadImage(input_path)
        print(f"   原始图像: size={original_image.GetSize()}, spacing={[f'{s:.2f}' for s in original_image.GetSpacing()]}")

        temp_output_path = os.path.join(self.config.TEMP_DIR, f"{case_name}.nii.gz")

        print("\n🎯 单阶段分割（全图，无裁剪）...")
        t0 = time.time()
        self.predictor.predict_from_files(
            [[input_path]],
            [temp_output_path],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )
        elapsed = time.time() - t0
        self.timings.append((case_name, elapsed))
        print(f"   分割完成，耗时 {elapsed:.2f}s")

        prediction = sitk.ReadImage(temp_output_path)

        # ========== 后处理：移除小标签 ==========
        print("\n🧹 后处理: 过滤小标签...")
        prediction = remove_small_labels(
            prediction,
            min_volume_cm3=self.config.MIN_LABEL_VOLUME_CM3
        )

        # 保存最终结果
        output_path = os.path.join(self.config.OUTPUT_DIR, f"{case_name}.nii.gz")
        sitk.WriteImage(prediction, output_path)

        print(f"   ✅ 保存: {output_path}")
        print(f"   最终预测: size={prediction.GetSize()}")

        # 清理临时文件
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

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

            existing_output = os.path.join(self.config.OUTPUT_DIR, f"{case_name}.nii.gz")
            if os.path.exists(existing_output):
                print(f"⏭️  跳过 {case_name}（已有结果）")
                results["success"].append(case_name)
                continue

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

        if self.timings:
            times = [t for _, t in self.timings]
            print(f"\n⏱️  推理耗时统计 (n={len(times)}):")
            print(f"   平均: {np.mean(times):.2f}s  标准差: {np.std(times):.2f}s")
            print(f"   最小: {np.min(times):.2f}s  最大: {np.max(times):.2f}s")

        return results


# ==================== 主函数 ====================
def main():
    print("🚀 单阶段脊椎分割推理（无裁剪，对照组）")
    print("   直接在全图上做分割，不经过 Stage 1 定位裁剪\n")

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
    predictor = SingleStagePredictor(Config)
    predictor.initialize()

    # 批量预测
    predictor.predict_batch(Config.INPUT_DIR)


if __name__ == "__main__":
    main()
