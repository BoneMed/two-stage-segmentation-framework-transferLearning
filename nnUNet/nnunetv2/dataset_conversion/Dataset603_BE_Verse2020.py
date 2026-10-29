"""  将除了0以为的lable转1  """

import numpy as np
import os
import cv2
import nibabel as nib

def convert_labels_to_binary(input_folder: str) -> None:
    """
    将输入文件夹中的nii标签文件转换为二值标签文件，除了0以外的标签都转为1。并覆盖文件

    参数:
        input_folder (str): 输入标签文件夹路径。
        output_folder (str): 输出二值标签文件夹路径。
    """

    for filename in os.listdir(input_folder):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            file_path = os.path.join(input_folder, filename)
            # 读取nii文件
            nii_img = nib.load(file_path)
            img_data = nii_img.get_fdata()
            # 转换为二值标签
            binary_data = np.where(img_data > 0, 1, 0).astype(np.uint8)
            # 保存覆盖原文件
            binary_nii = nib.Nifti1Image(binary_data, affine=nii_img.affine, header=nii_img.header)
            nib.save(binary_nii, file_path)
            print(f'{filename} converted')


if __name__ == '__main__':
    input_folder = r"N:\ThirdYear\project0121\Data\nnUNet_raw\Dataset101_verseg_BE\labelsTr"  # 替换为你的标签文件夹路径
    convert_labels_to_binary(input_folder)

