"""
医学图像三维可视化 - 椎骨Mask渲染（双窗口独立版）
"""

import vtk
import numpy as np
import nibabel as nib
from vtk.util import numpy_support
import threading

# ============ 椎骨标签和颜色定义 ============
v_dict = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

colors_itk = (1 / 255) * np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    [255, 0, 255], [255, 239, 213],
    [0, 0, 205], [205, 133, 63], [210, 180, 140], [102, 205, 170], [0, 0, 128],
    [0, 139, 139], [46, 139, 87], [255, 228, 225], [106, 90, 205], [221, 160, 221],
    [233, 150, 122], [165, 42, 42],
    [255, 250, 250], [147, 112, 219], [218, 112, 214], [75, 0, 130], [255, 182, 193],
    [60, 179, 113], [255, 235, 205],
    [255, 235, 205], [255, 228, 196],
])


class SpineWindow:
    """单个椎骨可视化窗口"""

    def __init__(self, title="Spine Visualization", position=(100, 100)):
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 800)
        self.render_window.SetWindowName(title)
        self.render_window.SetPosition(position[0], position[1])

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

    def load_nifti(self, filepath):
        """加载NIfTI格式的医学图像"""
        nifti_img = nib.load(filepath)
        data = nifti_img.get_fdata()
        spacing = nifti_img.header.get_zooms()[:3]
        return data, spacing

    def numpy_to_vtk_image(self, numpy_array, spacing=(1.0, 1.0, 1.0)):
        """将NumPy数组转换为VTK图像数据"""
        if numpy_array.dtype != np.float32:
            numpy_array = numpy_array.astype(np.float32)

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(numpy_array.shape)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(0, 0, 0)

        flat_array = numpy_array.flatten(order='F')
        vtk_array = numpy_support.numpy_to_vtk(flat_array, deep=True)
        vtk_image.GetPointData().SetScalars(vtk_array)

        return vtk_image

    def create_surface_mesh(self, mask_data, spacing, threshold=0.5):
        """从mask数据创建表面网格"""
        vtk_image = self.numpy_to_vtk_image(mask_data, spacing)

        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(vtk_image)
        marching_cubes.SetValue(0, threshold)
        marching_cubes.ComputeNormalsOn()
        marching_cubes.Update()

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(marching_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(50)
        smoother.SetPassBand(0.01)
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.Update()

        return normals.GetOutput()

    def add_surface_actor(self, poly_data, color=(0.9, 0.85, 0.75), opacity=0.9):
        """添加表面渲染的Actor"""
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetAmbient(0.2)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)

        self.renderer.AddActor(actor)
        return actor

    def add_multi_label_surface(self, mask_data, spacing):
        """为多标签mask添加不同颜色的表面渲染"""
        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels > 0]

        for label in unique_labels:
            label_int = int(label)
            if label_int <= len(colors_itk):
                color = tuple(colors_itk[label_int - 1])
            else:
                color = (0.8, 0.8, 0.8)

            binary_mask = (mask_data == label).astype(np.float32)
            poly_data = self.create_surface_mesh(binary_mask, spacing, threshold=0.5)
            self.add_surface_actor(poly_data, color=color, opacity=0.9)

    def add_text_annotation(self, text, position=(0.05, 0.92)):
        """添加文字标注"""
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(text)
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor.GetPositionCoordinate().SetValue(position[0], position[1])

        text_property = text_actor.GetTextProperty()
        text_property.SetFontSize(24)
        text_property.SetColor(1.0, 1.0, 1.0)
        text_property.SetFontFamilyToArial()
        text_property.BoldOn()

        self.renderer.AddActor2D(text_actor)

    def setup(self, bg_color1=(0.1, 0.1, 0.2), bg_color2=(0.3, 0.3, 0.4)):
        """设置渲染器"""
        self.renderer.SetBackground(bg_color1)
        self.renderer.SetBackground2(bg_color2)
        self.renderer.GradientBackgroundOn()

        # 添加光照
        self.renderer.RemoveAllLights()

        key_light = vtk.vtkLight()
        key_light.SetFocalPoint(0, 0, 0)
        key_light.SetPosition(100, 100, 100)
        key_light.SetColor(1.0, 1.0, 1.0)
        key_light.SetIntensity(1.0)
        self.renderer.AddLight(key_light)

        fill_light = vtk.vtkLight()
        fill_light.SetFocalPoint(0, 0, 0)
        fill_light.SetPosition(-100, 50, 50)
        fill_light.SetColor(0.9, 0.9, 1.0)
        fill_light.SetIntensity(0.5)
        self.renderer.AddLight(fill_light)

    def setup_camera(self):
        """设置相机视角"""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Elevation(-90)
        camera.Azimuth(30)
        camera.Roll(180)
        self.renderer.ResetCameraClippingRange()

    def start(self):
        """启动渲染"""
        self.render_window.Render()
        self.interactor.Initialize()
        self.interactor.Start()


def compare_spines_dual_window(mask_path1, mask_path2,
                               title1="Ground Truth", title2="Prediction"):
    """
    使用双独立窗口对比两组椎骨数据
    """
    # 创建两个独立窗口，设置不同位置
    window1 = SpineWindow(title=title1, position=(50, 100))
    window2 = SpineWindow(title=title2, position=(900, 100))

    # 加载数据
    print("正在加载数据...")
    mask_data1, spacing1 = window1.load_nifti(mask_path1)
    mask_data2, spacing2 = window2.load_nifti(mask_path2)

    print(f"数据1: {mask_data1.shape}, 标签: {np.unique(mask_data1)}")
    print(f"数据2: {mask_data2.shape}, 标签: {np.unique(mask_data2)}")

    # 设置窗口1
    window1.setup()
    window1.add_multi_label_surface(mask_data1, spacing1)
    window1.add_text_annotation(title1)
    window1.setup_camera()

    # 设置窗口2
    window2.setup()
    window2.add_multi_label_surface(mask_data2, spacing2)
    window2.add_text_annotation(title2)
    window2.setup_camera()

    print("启动可视化窗口...")
    print("两个窗口可独立操作，按 'q' 关闭各自窗口")

    # 先渲染两个窗口
    window1.render_window.Render()
    window2.render_window.Render()

    # 初始化交互器
    window1.interactor.Initialize()
    window2.interactor.Initialize()

    window1.interactor.Start()


# ============ 使用示例 ============
if __name__ == "__main__":
    mask_path1 = "/mnt/PhD/pythonproject/PHD_PROGRAM/project0121/Data/nnUNet_raw/Dataset601_verseg/labelsTs_pred_FENet_trans_0.005/sub_gl124.nii.gz"
    mask_path2 = "/mnt/PhD/pythonproject/PHD_PROGRAM/project0121/Data/nnUNet_raw/Dataset601_verseg/labelsTs_pred_nnUNet/sub_gl124.nii.gz"

    compare_spines_dual_window(
        mask_path1, mask_path2,
        title1="FE",
        title2="NN"
    )