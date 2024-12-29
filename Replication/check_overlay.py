import numpy as np
from PIL import Image
import random
import cv2

# 读取拼接图像
image_path = "/root/autodl-tmp/cv-illusion/output: 1_1735471263/ims_0049.png"
combined_image = cv2.imread(image_path)

# 获取拼接图像的大小
height, width, _ = combined_image.shape

# 定义行列数
rows = 2
cols = 4

# 计算每个子图像的大小
sub_width = width // cols
sub_height = height // rows

# 裁剪并保存子图像
for i in range(rows):
    for j in range(cols):
        # 计算子图像的左上角和右下角坐标
        left = j * sub_width
        top = i * sub_height
        right = left + sub_width
        bottom = top + sub_height

        # 裁剪子图像
        sub_image = combined_image[top:bottom, left:right]

        # 保存子图像
        cv2.imwrite(f"/root/autodl-tmp/cv-illusion/output: 1_1735471263/sub_image_{i}_{j}.png", sub_image)

print("子图像已保存！")

# 模拟叠加函数
def simulate_overlay(bottom, top, clean_mode=True):
    if clean_mode:
        exp = 1
        brightness = 3
        black = 0
    else:
        exp = random.uniform(0.5, 1)
        brightness = random.uniform(1, 5)
        black = random.uniform(0, 0.5)
        bottom = blend(bottom, black, random.random())
        top = blend(top, black, random.random())
    
    # 叠加公式
    result = (bottom**exp * top**exp * brightness)
    # 限制像素值范围
    result = np.clip(result, 0, 99)
    # 双曲正切变换
    result = np.tanh(result)
    return result

# 图像混合函数
def blend(image, black, alpha):
    return image * (1 - alpha) + black * alpha

# 图像旋转函数
def rotate_image(image, k):
    # 旋转图像
    rotated = np.rot90(image, k=k, axes=(0, 1))  # 在高度和宽度维度上旋转
    return rotated

# 获取可学习图像
def get_learnable_image(bottom, top, rotation):
    top_rotated = rotate_image(top, rotation)
    return simulate_overlay(bottom, top_rotated)

# 示例：加载图像
bottom_image = np.array(Image.open("/root/autodl-tmp/cv-illusion/output: 1_1735471263/sub_image_1_0.png").convert("RGB")) / 255.0
top_image = np.array(Image.open("/root/autodl-tmp/cv-illusion/output: 1_1735471263/sub_image_1_1.png").convert("RGB")) / 255.0

# 生成不同旋转角度的叠加图像
learnable_image_w = get_learnable_image(bottom_image, top_image, 0)
learnable_image_x = get_learnable_image(bottom_image, top_image, 1)
learnable_image_y = get_learnable_image(bottom_image, top_image, 2)
learnable_image_z = get_learnable_image(bottom_image, top_image, 3)

# 将结果保存为图像
def save_image(array, filename):
    array = (array * 255).astype(np.uint8)
    Image.fromarray(array).save(filename)

save_image(learnable_image_w, "learnable_image_w.png")
save_image(learnable_image_x, "learnable_image_x.png")
save_image(learnable_image_y, "learnable_image_y.png")
save_image(learnable_image_z, "learnable_image_z.png")

print("叠加图像已保存！")