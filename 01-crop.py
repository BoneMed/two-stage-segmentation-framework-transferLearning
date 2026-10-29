import os
import SimpleITK as sitk
import numpy as np
import torch
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def resample_img(itk_image, out_spacing=[3.0, 3.0, 6.0], is_label=False):
    """重采样图像到指定spacing"""
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    
    return resample.Execute(itk_image)


def crop_xyz_axis(image, low_res_segmentation, margin=50):
    """
    裁剪X、Y、Z三个轴    
    Parameters:
    -----------
    image : sitk.Image
        原始高分辨率图像
    low_res_segmentation : sitk.Image
        低分辨率分割mask
    margin_y : float
        Y轴方向的边距(mm)
    
    Returns:
    --------
    cropped_image : sitk.Image
        裁剪后的图像
    crop_coordinates : dict
        裁剪坐标
    """
    # 获取mask的numpy数组
    mask_np = sitk.GetArrayFromImage(low_res_segmentation)
    
    # 找到mask非零区域的边界
    nonzero_indices = np.nonzero(mask_np)
    
    if len(nonzero_indices[0]) == 0:
        raise ValueError("Mask is empty, no foreground found!")
    
   # 获取X、Y、Z轴的边界
    # numpy数组形状是(z, y, x)
    min_z = int(min(nonzero_indices[0]))
    max_z = int(max(nonzero_indices[0]))
    min_y = int(min(nonzero_indices[1]))
    max_y = int(max(nonzero_indices[1]))
    min_x = int(min(nonzero_indices[2]))
    max_x = int(max(nonzero_indices[2]))
    
    # 转换到物理坐标
    # SimpleITK的索引顺序是(x, y, z)
    min_point_physical = low_res_segmentation.TransformIndexToPhysicalPoint((min_x, min_y, min_z))
    max_point_physical = low_res_segmentation.TransformIndexToPhysicalPoint((max_x, max_y, max_z))
    
    # 映射回原始图像的索引
    min_point_orig = image.TransformPhysicalPointToIndex(min_point_physical)
    max_point_orig = image.TransformPhysicalPointToIndex(max_point_physical)
    
    # 获取原始图像信息
    spacing = image.GetSpacing()
    size = image.GetSize()
    
    # 计算各轴的margin（以像素为单位）
    margin_x_pixels = int(margin / spacing[0])
    margin_y_pixels = int(margin / spacing[1])
    margin_z_pixels = int(margin / spacing[2])
    
    # 裁剪X、Y、Z三个轴
    x_start = max(0, min_point_orig[0] - margin_x_pixels)
    x_finish = min(size[0], max_point_orig[0] + margin_x_pixels)

     # 修改这两行：从固定值改为根据mask裁剪
    y_start = max(0, min_point_orig[1] - margin_y_pixels)      # 原来是 y_start = 0
    y_finish = min(size[1], max_point_orig[1] + margin_y_pixels)  # 原来是 y_finish = size[1]
    
    z_start = max(0, min_point_orig[2] - margin_z_pixels)
    z_finish = min(size[2], max_point_orig[2] + margin_z_pixels)
    
    # 执行裁剪
    cropped_image = image[x_start:x_finish, y_start:y_finish, z_start:z_finish]
    
    crop_coordinates = {
        'x_start': x_start,
        'x_finish': x_finish,
        'y_start': y_start,
        'y_finish': y_finish,
        'z_start': z_start,
        'z_finish': z_finish
    }
    
    return cropped_image, crop_coordinates


def initialize_nnunet_predictor(model_path, device='cuda'):
    """初始化nnUNet预测器"""
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=('0',),
        checkpoint_name='checkpoint_best.pth',
    )
    
    return predictor


def run_nnunet_inference_single(predictor, input_path, output_dir, case_name):
    """单模态nnUNet推理"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{case_name}.nii.gz")
    
    # 准备输入文件（单模态）
    predictor.predict_from_files(
        [[input_path]],  # 单模态输入
        [output_path],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    
    return output_path

    
def main():
    # ==================== 配置路径 ====================
    # 输入数据路径
    input_images_dir = "/root/autodl-tmp/Data/nnUNet_raw/Dataset601_verseg/imagesTr"
    input_labels_dir = "/root/autodl-tmp/Data/nnUNet_raw/Dataset601_verseg/labelsTr"
    
    # nnUNet脊柱分割模型路径
    spine_model_path = "/root/autodl-tmp/Data/nnUNet_results/Dataset101_verseg/nnUNetTrainer__nnUNetPlans__3d_fullres_my_spacing"
    
    # 临时文件目录
    temp_dir = "/root/autodl-tmp/temp_spine_seg"
    
    # 输出目录
    output_base_dir = "/root/autodl-tmp/Data/nnUNet_raw/Dataset602_verseg_cropped"
    output_images_dir = os.path.join(output_base_dir, "imagesTr")
    output_labels_dir = os.path.join(output_base_dir, "labelsTr")
    
    # Y轴裁剪边距(mm)
    margin = 30
    
    # ==================== 创建目录 ====================
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # ==================== 初始化模型 ====================
    print("初始化nnUNet脊柱分割模型...")
    predictor = initialize_nnunet_predictor(spine_model_path)
    print("模型初始化成功!")
    
    # ==================== 获取所有图像文件 ====================
    image_files = [f for f in os.listdir(input_images_dir) if f.endswith('.nii.gz')]
    print(f"找到 {len(image_files)} 个图像文件")
    
    # ==================== 处理每个图像 ====================
    for image_file in image_files:
        # 获取case名称
        case_name = image_file.replace('.nii.gz', '').replace('_0000', '')
        print(f"\n{'='*50}")
        print(f"处理: {case_name}")
        
        image_path = os.path.join(input_images_dir, image_file)
        
        # Step 1: 读取原始图像
        original_image = sitk.ReadImage(image_path)
        print(f"原始图像: size={original_image.GetSize()}, spacing={original_image.GetSpacing()}")
        
        # Step 2: 重采样到低分辨率用于分割
        low_res_spacing = [6.0, 6.0, 6.0]  # 根据你训练模型时的spacing调整
        low_res_image = resample_img(original_image, out_spacing=low_res_spacing)
        print(f"低分辨率图像: size={low_res_image.GetSize()}")
        
        # 保存临时文件用于推理
        temp_input_path = os.path.join(temp_dir, f"{case_name}_0000.nii.gz")
        sitk.WriteImage(low_res_image, temp_input_path)
        
        # Step 3: 运行nnUNet分割
        print("运行脊柱分割...")
        seg_output_path = run_nnunet_inference_single(
            predictor, temp_input_path, temp_dir, case_name
        )
        
        # 读取分割结果
        spine_mask = sitk.ReadImage(seg_output_path)
        spine_mask_arr = sitk.GetArrayFromImage(spine_mask)
        
        # 二值化（如果模型输出多标签）
        spine_mask_arr[spine_mask_arr > 0] = 1
        spine_mask_binary = sitk.GetImageFromArray(spine_mask_arr.astype(np.uint8))
        spine_mask_binary.CopyInformation(spine_mask)
        
        print(f"分割mask唯一值: {np.unique(spine_mask_arr)}")
        
        # 检查是否找到脊柱
        if 1 not in np.unique(spine_mask_arr):
            print(f"警告: 未找到脊柱区域，跳过 {case_name}")
            continue
        
        # Step 4: 只裁剪Y轴
        print(f"裁剪XYZ轴 (margin={margin}mm)...")
        try:
            cropped_image, crop_coords = crop_xyz_axis(
                original_image, spine_mask_binary, margin=margin
            )
            print(f"裁剪后图像: size={cropped_image.GetSize()}")
            print(f"裁剪坐标: X={crop_coords['x_start']}:{crop_coords['x_finish']}, Y={crop_coords['y_start']}:{crop_coords['y_finish']}, Z={crop_coords['z_start']}:{crop_coords['z_finish']}")
        except Exception as e:
            print(f"裁剪失败: {e}")
            continue
        
        # Step 5: 保存裁剪后的图像
        output_image_path = os.path.join(output_images_dir, f"{case_name}_0000.nii.gz")
        sitk.WriteImage(cropped_image, output_image_path)
        print(f"保存: {output_image_path}")
        
        # Step 6: 如果有标签，也进行相同的裁剪
        label_path = os.path.join(input_labels_dir, f"{case_name}.nii.gz")
        if os.path.exists(label_path):
            original_label = sitk.ReadImage(label_path)
            cropped_label = original_label[
                crop_coords['x_start']:crop_coords['x_finish'],
                crop_coords['y_start']:crop_coords['y_finish'],
                crop_coords['z_start']:crop_coords['z_finish']
            ]
            output_label_path = os.path.join(output_labels_dir, f"{case_name}.nii.gz")
            sitk.WriteImage(cropped_label, output_label_path)
            print(f"保存标签: {output_label_path}")
        
        # 清理临时文件
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(seg_output_path):
            os.remove(seg_output_path)
    
    print(f"\n{'='*50}")
    print("处理完成!")
    print(f"裁剪后的图像保存在: {output_images_dir}")
    print(f"裁剪后的标签保存在: {output_labels_dir}")


if __name__ == '__main__':
    main()