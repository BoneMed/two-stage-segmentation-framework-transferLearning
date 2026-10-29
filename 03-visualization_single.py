"""
医学图像三维可视化 - 椎骨Mask渲染
使用VTK进行三维可视化渲染
"""

import vtk
import numpy as np
import nibabel as nib
from vtk.util import numpy_support

# ============ 添加椎骨标签和颜色定义 ============
v_dict = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

colors_itk = (1/255)*np.array([
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # Label 1-7 (C1-7)
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122], [165, 42, 42],  # Label 8-19 (T1-12)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205],  # Label 20-26 (L1-6, sacrum)
    [255,235,205], [255,228,196],  # Label 27 cocc, 28 T13
])


class SpineVisualizer:
    """椎骨三维可视化器"""

    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1200, 800)
        self.render_window.SetWindowName("椎骨三维可视化")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # 设置交互样式
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

    def load_nifti(self, filepath):
        """加载NIfTI格式的医学图像"""
        nifti_img = nib.load(filepath)
        data = nifti_img.get_fdata()
        affine = nifti_img.affine
        spacing = nifti_img.header.get_zooms()[:3]
        return data, affine, spacing

    def numpy_to_vtk_image(self, numpy_array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        """将NumPy数组转换为VTK图像数据"""
        # 确保数据类型正确
        if numpy_array.dtype != np.float32:
            numpy_array = numpy_array.astype(np.float32)

        # 创建VTK图像数据
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(numpy_array.shape)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(origin)

        # 将NumPy数组转换为VTK数组
        flat_array = numpy_array.flatten(order='F')  # Fortran顺序以匹配VTK
        vtk_array = numpy_support.numpy_to_vtk(flat_array, deep=True)
        vtk_image.GetPointData().SetScalars(vtk_array)

        return vtk_image

    def create_surface_mesh(self, mask_data, spacing, threshold=0.5):
        """从mask数据创建表面网格（Marching Cubes算法）"""
        vtk_image = self.numpy_to_vtk_image(mask_data, spacing)

        # 使用Marching Cubes提取等值面
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(vtk_image)
        marching_cubes.SetValue(0, threshold)
        marching_cubes.ComputeNormalsOn()
        marching_cubes.Update()

        # 使用Windowed Sinc平滑（比Laplacian平滑效果更好，保持体积）
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(marching_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(50)  # 增加迭代次数
        smoother.SetPassBand(0.01)  # 控制平滑程度，值越小越平滑
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        # 计算法向量以获得更好的光照效果
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.Update()

        return normals.GetOutput()

    def add_surface_actor(self, poly_data, color=(0.9, 0.85, 0.75), opacity=1.0):
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

    def add_volume_rendering(self, image_data, spacing, color_func=None, opacity_func=None):
        """添加体渲染"""
        vtk_image = self.numpy_to_vtk_image(image_data, spacing)

        # 创建颜色传递函数
        if color_func is None:
            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(-1000, 0.0, 0.0, 0.0)  # 空气 - 黑色
            color_func.AddRGBPoint(-200, 0.4, 0.2, 0.1)  # 软组织
            color_func.AddRGBPoint(200, 0.9, 0.6, 0.5)  # 肌肉
            color_func.AddRGBPoint(500, 0.95, 0.85, 0.7)  # 骨骼
            color_func.AddRGBPoint(1500, 1.0, 1.0, 0.9)  # 致密骨

        # 创建不透明度传递函数
        if opacity_func is None:
            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(-1000, 0.0)
            opacity_func.AddPoint(-200, 0.0)
            opacity_func.AddPoint(200, 0.1)
            opacity_func.AddPoint(500, 0.4)
            opacity_func.AddPoint(1500, 0.8)

        # 梯度不透明度
        gradient_opacity = vtk.vtkPiecewiseFunction()
        gradient_opacity.AddPoint(0, 0.0)
        gradient_opacity.AddPoint(90, 0.5)
        gradient_opacity.AddPoint(100, 1.0)

        # 体属性
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.SetGradientOpacity(gradient_opacity)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.4)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.2)

        # 体映射器（GPU加速）
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)
        volume_mapper.SetBlendModeToComposite()

        # 创建体
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        self.renderer.AddVolume(volume)
        return volume

    def add_multi_label_surface(self, mask_data, spacing, label_colors=None):
        """
        为多标签mask添加不同颜色的表面渲染
        适用于不同椎骨有不同标签的情况
        """
        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels > 0]  # 排除背景

        # 使用VerSe数据集定义的颜色映射
        if label_colors is None:
            label_colors = {}
            for label in unique_labels:
                label_int = int(label)
                if label_int <= len(colors_itk):
                    # colors_itk 索引从0开始，标签从1开始
                    label_colors[label] = tuple(colors_itk[label_int - 1])
                else:
                    label_colors[label] = (0.8, 0.8, 0.8)  # 默认灰色

        actors = []
        for label in unique_labels:
            # 创建单个标签的二值mask
            binary_mask = (mask_data == label).astype(np.float32)

            # 创建表面网格
            poly_data = self.create_surface_mesh(binary_mask, spacing, threshold=0.5)

            # 获取颜色
            color = label_colors.get(label, (0.8, 0.8, 0.8))

            # 添加Actor
            actor = self.add_surface_actor(poly_data, color=color, opacity=0.9)
            actors.append(actor)

            # 显示椎骨名称和颜色
            label_int = int(label)
            vertebra_name = v_dict.get(label_int, f"Unknown-{label_int}")
            print(f"已添加标签 {label_int} ({vertebra_name}), 颜色: RGB{tuple(int(c*255) for c in color)}")

        return actors

    def setup_camera(self):
        """设置相机视角"""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Elevation(-90)  # 调整视角
        camera.Azimuth(30)
        camera.Roll(180)
        self.renderer.ResetCameraClippingRange()

    def add_axes(self):
        """添加坐标轴"""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)

        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.interactor)
        widget.SetViewport(0.0, 0.0, 0.15, 0.15)
        widget.SetEnabled(1)
        widget.InteractiveOff()

        return widget

    def add_lighting(self):
        """添加光照"""
        # 移除默认光照
        self.renderer.RemoveAllLights()

        # 主光源
        key_light = vtk.vtkLight()
        key_light.SetFocalPoint(0, 0, 0)
        key_light.SetPosition(100, 100, 100)
        key_light.SetColor(1.0, 1.0, 1.0)
        key_light.SetIntensity(1.0)
        self.renderer.AddLight(key_light)

        # 补光
        fill_light = vtk.vtkLight()
        fill_light.SetFocalPoint(0, 0, 0)
        fill_light.SetPosition(-100, 50, 50)
        fill_light.SetColor(0.9, 0.9, 1.0)
        fill_light.SetIntensity(0.5)
        self.renderer.AddLight(fill_light)

        # 背光
        back_light = vtk.vtkLight()
        back_light.SetFocalPoint(0, 0, 0)
        back_light.SetPosition(0, -100, 100)
        back_light.SetColor(1.0, 0.95, 0.9)
        back_light.SetIntensity(0.3)
        self.renderer.AddLight(back_light)

    def set_background(self, color1=(0.1, 0.1, 0.2), color2=(0.3, 0.3, 0.4)):
        """设置渐变背景"""
        self.renderer.SetBackground(color1)
        self.renderer.SetBackground2(color2)
        self.renderer.GradientBackgroundOn()

    def add_text_annotation(self, text, position=(0.01, 0.95)):
        """添加文字标注"""
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(text)
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor.GetPositionCoordinate().SetValue(position[0], position[1])

        text_property = text_actor.GetTextProperty()
        text_property.SetFontSize(18)
        text_property.SetColor(1.0, 1.0, 1.0)
        text_property.SetFontFamilyToArial()
        text_property.BoldOn()

        self.renderer.AddActor2D(text_actor)
        return text_actor

    def save_screenshot(self, filename="spine_visualization.png"):
        """保存截图"""
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.render_window)
        window_to_image.SetScale(2)  # 2x分辨率
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        print(f"截图已保存: {filename}")

    def start(self):
        """启动交互式渲染"""
        self.render_window.Render()
        self.interactor.Start()


def visualize_spine(image_path, mask_path, render_mode="surface"):
    """
    主函数：可视化椎骨

    参数:
        image_path: 原始医学图像路径 (.nii, .nii.gz)
        mask_path: 椎骨mask路径 (.nii, .nii.gz)
        render_mode: 渲染模式 - "surface"(表面渲染), "volume"(体渲染), "both"(两者)
    """
    visualizer = SpineVisualizer()

    # 加载数据
    print("正在加载数据...")
    image_data, image_affine, image_spacing = visualizer.load_nifti(image_path)
    mask_data, mask_affine, mask_spacing = visualizer.load_nifti(mask_path)

    print(f"图像尺寸: {image_data.shape}")
    print(f"图像spacing: {image_spacing}")
    print(f"Mask尺寸: {mask_data.shape}")
    print(f"Mask中的唯一标签: {np.unique(mask_data)}")

    # 设置背景和光照
    visualizer.set_background()
    #visualizer.set_background(color1=(1.0, 1.0, 1.0), color2=(0.95, 0.95, 0.95))
    visualizer.add_lighting()

    if render_mode in ["surface", "both"]:
        print("正在创建表面渲染...")
        # 检查是否是多标签mask
        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) > 1:
            # 多标签：为每个椎骨使用不同颜色
            visualizer.add_multi_label_surface(mask_data, mask_spacing)
        else:
            # 单标签：使用骨骼颜色
            binary_mask = (mask_data > 0).astype(np.float32)
            poly_data = visualizer.create_surface_mesh(binary_mask, mask_spacing)
            visualizer.add_surface_actor(poly_data, color=(0.95, 0.87, 0.73), opacity=0.9)

    if render_mode in ["volume", "both"]:
        print("正在创建体渲染...")
        # 使用mask裁剪原始图像，只显示椎骨区域
        masked_image = np.where(mask_data > 0, image_data, -1000)
        visualizer.add_volume_rendering(masked_image, image_spacing)

    # 添加坐标轴和标注
    visualizer.add_axes()
    visualizer.add_text_annotation("Spine Visualization - VTK")

    # 设置相机
    visualizer.setup_camera()

    print("启动可视化窗口...")
    print("交互说明:")
    print("  - 左键拖动: 旋转")
    print("  - 右键拖动: 缩放")
    print("  - 中键拖动: 平移")
    print("  - 按 'q' 退出")

    visualizer.start()


# ============ 使用示例 ============
if __name__ == "__main__":
    # 方式1：直接指定文件路径
    image_path = "nanjing_20170121000036_0000.nii.gz"  # 替换为你的CT图像路径
    mask_path = "nanjing_20170121000036.nii.gz"  # 替换为你的椎骨mask路径

    # 运行可视化
    # render_mode可选: "surface"(表面渲染), "volume"(体渲染), "both"(两者)
    visualize_spine(image_path, mask_path, render_mode="surface")