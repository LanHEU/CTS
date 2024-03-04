'''
Author: ZhangLan
Date: 2024-03-02 12:34:48
LastEditors: ZhangLan
LastEditTime: 2024-03-02 14:43:44
Description: file content
'''
import numpy as np
import nibabel as nib

def remove_slices_without_label(img_path, img1_path, img2_path, img3_path, label_path):
    # 加载图像和标签
    img = nib.load(img_path)
    img1 = nib.load(img1_path)
    img2 = nib.load(img2_path)
    img3 = nib.load(img3_path)
    label = nib.load(label_path)

    # 获取图像和标签的形状
    img_shape = img.shape
    print(img_shape)
    label_shape = label.shape
    print(label_shape)

    # 确保图像和标签具有相同的大小
    assert img_shape == label_shape, "Image and label shapes must be the same."

    # 初始化一个布尔数组，用于存储要保留的切片索引
    keep_slices = np.zeros(img_shape[2], dtype=bool)

    # 遍历图像的切片，检查它们是否具有标签
    # for slice_idx in range(img_shape[2]):
    #     img_slice = img.get_fdata()[:, :, slice_idx]
    #     label_slice = label.get_fdata()[:, :, slice_idx]
    #     a=np.sum(np.sum(label_slice))
    #     # 检查切片是否具有至少一个非零标签值
    #     if a > 0:
    #         keep_slices[slice_idx] = True
    #         print(slice_idx)
    # 遍历图片前8个
    for slice_idx in range(42,50):
        img_slice = img.get_fdata()[:, :, slice_idx]
        label_slice = label.get_fdata()[:, :, slice_idx]
        
        keep_slices[slice_idx] = True


    # 根据布尔数组保留切片
    img_data = img.get_fdata()[:, :, keep_slices]
    img1_data = img1.get_fdata()[:, :, keep_slices]
    img2_data = img2.get_fdata()[:, :, keep_slices]
    img3_data = img3.get_fdata()[:, :, keep_slices]
    label_data = label.get_fdata()[:, :, keep_slices]

    print (img_data.shape)
    print (label_data.shape)

    # 创建一个新的Nifti1图像并保存结果
    new_img = nib.Nifti1Image(img_data, img.affine)
    new_img1 = nib.Nifti1Image(img1_data, img1.affine)
    new_img2 = nib.Nifti1Image(img2_data, img2.affine)
    new_img3 = nib.Nifti1Image(img3_data, img3.affine)
    new_label = nib.Nifti1Image(label_data, label.affine)

    nib.save(new_img, "D:/data/Brast/HGG/c/c/flair.nii.gz")
    nib.save(new_img1, "D:/data/Brast/HGG/c/c/t1.nii.gz")
    nib.save(new_img2, "D:/data/Brast/HGG/c/c/t1ce.nii.gz")
    nib.save(new_img3, "D:/data/Brast/HGG/c/c/t2.nii.gz")
    nib.save(new_label, "D:/data/Brast/HGG/c/c/seg.nii.gz")

# 使用示例
img_path = "D:/data/Brast/HGG/a/flair.nii.gz"
img1_path = "D:/data/Brast/HGG/a/t1.nii.gz"
img2_path = "D:/data/Brast/HGG/a/t1ce.nii.gz"
img3_path = "D:/data/Brast/HGG/a/t2.nii.gz"
label_path = "D:/data/Brast/HGG/a/seg.nii.gz"
remove_slices_without_label(img_path, img1_path, img2_path, img3_path, label_path)
