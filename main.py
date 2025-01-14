import gradio as gr
import yaml
from PIL import Image
import numpy as np
import time
import os
import torch
import torch.nn as nn
import source.stable_diffusion as sd
from easydict import EasyDict
from source.learnable_textures import LearnableImageFourier
from source.stable_diffusion_labels import NegativeLabel
from itertools import chain
import rp
from diffusers import StableDiffusionPipeline
from amzqr import amzqr
import qrcode
import random
from torchvision import transforms
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image, make_image_grid

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from huggingface_hub import login
import subprocess

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

token = "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"
login(token)
print("log in successfully.")

# 加载示例 Prompts
def load_example_prompts(file_path="source/example_prompts.yaml"):
    """
    从 YAML 文件加载示例 Prompts。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        return "\n".join(f"- {value}" for value in prompts.values())
    except Exception as e:
        print(f"加载YAML文件出错: {e}")
        return ""


generating_flag = True

def generate_parker_puzzle(prompt_a, prompt_b, negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4, model_name="CompVis/stable-diffusion-v1-4"):
    combined_prompt = f"{prompt_a}\n{prompt_b}"
    print('prompt a: ', prompt_a)
    print('prompt b: ', prompt_b)
    print('negative prompt: ', negative_prompt)
    gpu = torch.device('cuda')
    if 's' not in dir():
        s = sd.StableDiffusion(gpu, model_name)
    device = s.device
    label_a = NegativeLabel(prompt_a, negative_prompt)
    label_b = NegativeLabel(prompt_b, negative_prompt)
    learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, num_features=256, hidden_dim=256, scale=10).to(s.device)
    image = learnable_image_maker()
    uv_map_b = rp.load_image("improvement1-mask/parker_puzzle_uv_map.png")
    uv_map_a = rp.get_identity_uv_map(*rp.get_image_dimensions(uv_map_b))
    
    learnable_image_a = lambda: rp.apply_uv_map(image(), uv_map_a)
    learnable_image_b = lambda: rp.apply_uv_map(image(), uv_map_b)
    
    optim = torch.optim.SGD(image.parameters(), lr=learning_rate)
    labels = [label_a, label_b]
    learnable_images = [learnable_image_a, learnable_image_b]
    weights = rp.as_numpy_array([1, 1.5])
    weights = weights / weights.sum() * len(weights)
    
    def get_display_image(border_color=(255, 255, 255)):
        return rp.tiled_images(
            [
                rp.as_numpy_image(learnable_image_a()),
                rp.as_numpy_image(learnable_image_b()),
            ],
            length=len(learnable_images),
            border_thickness=2,
            border_color=border_color
        )
    global generating_flag
    generating_flag = True

    try:
        for iter_num in range(num_iter+1):
            if not generating_flag:
                break
            preds = []
            for label, learnable_image, weight in rp.random_batch(list(zip(labels, learnable_images, weights)), batch_size=1):
                pred = s.train_step(
                    label.embedding,
                    learnable_image()[None],
                    noise_coef=0.1 * weight, guidance_scale=100,
                )
                preds += list(pred)
            with torch.no_grad():
                if not iter_num % (display_interval // 4):
                    im = get_display_image()
                    if not iter_num % display_interval:
                        progress = int(iter_num / num_iter * 100)
                        yield im, f"进度：{progress}%\n使用的Prompt:\n{combined_prompt}"
            optim.step()
            optim.zero_grad()
    except Exception as e:
        print(f"Error occurred: {e}")
        yield None, f"生成失败: {str(e)}"
    except not generating_flag:
        print("\nInterrupted early.")
        im = get_display_image()
        yield im, f"生成中断！\n使用的Prompt:\n{combined_prompt}"

iter = 0
def generate_random_puzzle(prompt_a, prompt_b, negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4, model_name="CompVis/stable-diffusion-v1-4"):
    global iter
    N = 4
    block = 64
    iter = 0
    def load_image(path):
        image = Image.open(path)
        rgb_image = image.convert('RGB')
        return np.array(rgb_image)

    def swap_cube(uv,gx,gy,fx,fy,size):
        global iter
        if abs(gx - fx) < size and abs(gy - fy) < size:
            return
        if gx + size > N or gy + size >  N or fx + size > N or fy + size > N:
            return
        iter += 1
        temp = np.copy(uv[gy * block:(gy + size) * block, gx * block:(gx + size) * block])
        uv[gy * block:(gy + size) * block, gx * block:(gx + size) * block] = np.copy(uv[fy * block:(fy + size) * block, fx * block:(fx + size) * block])
        uv[fy * block:(fy + size) * block, fx * block:(fx + size) * block] = temp
        if random.choice([True, False]):
            uv[gy * block:(gy + size) * block, gx * block:(gx + size) * block] = np.transpose(uv[gy * block:(gy + size) * block, gx * block:(gx + size) * block],(1,0,2))
        if random.choice([True, False]):
            uv[fy * block:(fy + size) * block, fx * block:(fx + size) * block] = np.transpose(uv[fy * block:(fy + size) * block, fx * block:(fx + size) * block],(1,0,2))

    def swap_triangle(uv,gx,gy,fx,fy,size):
        global iter
        if random.choice([True,False]):
            if abs(gx - fx) < size and abs(gy - fy) < size:
                return
            if gy - size < 0 or gx - size < 0 or gx + size > N or fy + size > N or fx - size < 0 or fx + size > N:
                return
            else:
                iter += 1
                for i in range(size * block):
                    gy_1 = gy * block - i - 1
                    fy_1 = fy * block + i
                    temp = np.copy(uv[gy_1:gy_1 + 1,gx * block - i:gx * block + i + 1,:])

                    uv[gy_1:gy_1 + 1,gx * block - i:gx * block + i + 1,:] = np.copy(uv[fy_1:fy_1 + 1,fx * block - i:fx * block + i + 1,:])
                    uv[fy_1:fy_1 + 1,fx * block - i:fx * block + i + 1,:] = temp
        else:
            if gy == fy and abs(gx - fx) < 2 * size:
                return
            if abs(gx - fx) < size and abs(gy - fy) < size:
                return
            if gy - size < 0 or gx - size < 0 or gx + size > N or fy - size < 0 or fx - size < 0 or fx + size > N:
                return
            else:
                iter += 1
                for i in range(size * block):
                    gy_1 = gy * block - i - 1
                    fy_1 = fy * block - i - 1
                    temp = np.copy(uv[gy_1:gy_1 + 1,gx * block - i:gx * block + i + 1,:])
                    uv[gy_1:gy_1 + 1,gx * block - i:gx * block + i + 1,:] = np.copy(uv[fy_1:fy_1 + 1,fx * block - i:fx * block + i + 1,:])
                    uv[fy_1:fy_1 + 1,fx * block - i:fx * block + i + 1,:] = temp
    def swap_triangle_row(uv,gx,gy,fx,fy,size):
        global iter
        if random.choice([True,False]):
            if gx - size < 0 or gy - size < 0 or gy + size > N or fx + size > N or fy - size < 0 or fy + size > N:
                return
            else:
                iter += 1
                for i in range(size * block):
                    gx_1 = gx * block - i - 1
                    fx_1 = fx * block + i
                    temp = np.copy(uv[gy * block - i:gy * block + i + 1,gx_1:gx_1 + 1,:])
                    uv[gy * block - i:gy * block + i + 1,gx_1:gx_1 + 1,:] = np.copy(uv[fy * block - i:fy * block + i + 1,fx_1:fx_1 + 1,:])
                    
                    uv[fy * block - i:fy * block + i + 1,fx_1:fx_1 + 1,:] = temp
        else:
            if gx == fx and abs(fy - fy) < 2 * size:
                return
            if abs(gx - fx) < size and abs(gy - fy) < size:
                return
            if gx - size < 0 or gy - size < 0 or gy + size > N or fx - size < 0 or fy - size < 0 or fy + size > N:
                return
            else:
                iter += 1
                for i in range(size * block):
                    gx_1 = gx * block - i - 1
                    fx_1 = fx * block - i - 1
                    temp = np.copy(uv[gy * block - i:gy * block + i + 1,gx_1:gx_1 + 1,:])
                    
                    uv[gy * block - i:gy * block + i + 1,gx_1:gx_1 + 1,:] = np.copy(uv[fy * block - i:fy * block + i + 1,fx_1:fx_1 + 1,:])
                    uv[fy * block - i:fy * block + i + 1,fx_1:fx_1 + 1,:] = temp

    def trans_uv(trans:int=10,unit:int=4):
        """

        :param trans:how many trans function will do
        :param unit: the min length of polygon, default is 4 means 1/4 photo size
        不妨先分几个情况分别决定要对什么样的进行替换就好了，

        :return:
        """
        global iter
        uv = load_image('improvement1-mask/uv_map_a.png')

        while iter < 2:
            gx = random.randint(0, N - 1)
            gy = random.randint(0, N - 1)
            fx = random.randint(0, N - 1)
            fy = random.randint(0, N - 1)
            size = random.random()
            if size < 0.1:
                size = 1
            else:
                size = 2
            swap_cube(uv,gx,gy,fx,fy,size)
        iter = 0
        while iter < 2:

            gx = random.randint(0, N)
            gy = random.randint(0, N)
            fx = random.randint(0, N)
            fy = random.randint(0, N)
            size = random.random()
            if size < 0.1:
                size = 1
            else:
                size = 2
            swap_triangle(uv,gx,gy,fx,fy,size)
        iter = 0
        while iter < 2:
            gx = random.randint(0, N)
            gy = random.randint(0, N)
            fx = random.randint(0, N)
            fy = random.randint(0, N)
            size = random.random()
            if size < 0.1:
                size = 1
            else:
                size = 2
            swap_triangle_row(uv,gx,gy,fx,fy,size)
        image = Image.fromarray(uv)
        image.save('improvement1-mask/trans_uv.png')

    trans_uv()

    combined_prompt = f"{prompt_a}\n{prompt_b}"
    print('prompt a: ', prompt_a)
    print('prompt b: ', prompt_b)
    print('negative prompt: ', negative_prompt)
    gpu = torch.device('cuda')
    if 's' not in dir():
        s = sd.StableDiffusion(gpu, model_name)
    
    device = s.device
    label_a = NegativeLabel(prompt_a, negative_prompt)
    label_b = NegativeLabel(prompt_b, negative_prompt)
    learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, num_features=256, hidden_dim=256, scale=10).to(s.device)
    image = learnable_image_maker()

    uv_map_b = rp.load_image("improvement1-mask/trans_uv.png")
    uv_map_a = rp.get_identity_uv_map(*rp.get_image_dimensions(uv_map_b))

    learnable_image_a = lambda: rp.apply_uv_map(image(), uv_map_a)
    learnable_image_b = lambda: rp.apply_uv_map(image(), uv_map_b)
    
    optim = torch.optim.SGD(image.parameters(), lr=learning_rate)
    labels = [label_a, label_b]
    learnable_images = [learnable_image_a, learnable_image_b]
    weights = rp.as_numpy_array([1, 1.5])
    weights = weights / weights.sum() * len(weights)
    
    def get_display_image(border_color=(255, 255, 255)):
        return rp.tiled_images(
            [
                rp.as_numpy_image(learnable_image_a()),
                rp.as_numpy_image(learnable_image_b()),
            ],
            length=len(learnable_images),
            border_thickness=2,
            border_color=border_color
        )
    
    global generating_flag
    generating_flag = True

    try:
        for iter_num in range(num_iter+1):
            if not generating_flag:
                break
            preds = []
            for label, learnable_image, weight in rp.random_batch(list(zip(labels, learnable_images, weights)), batch_size=1):
                pred = s.train_step(
                    label.embedding,
                    learnable_image()[None],
                    noise_coef=0.1 * weight, guidance_scale=100,
                )
                preds += list(pred)
            with torch.no_grad():
                if not iter_num % (display_interval // 4):
                    im = get_display_image()
                    if not iter_num % display_interval:
                        progress = int(iter_num / num_iter * 100)
                        yield im, f"进度：{progress}%\n使用的Prompt:\n{combined_prompt}"
            optim.step()
            optim.zero_grad()
    except Exception as e:
        print(f"Error occurred: {e}")
        yield None, f"生成失败: {str(e)}"
    except not generating_flag:
        print("\nInterrupted early.")
        im = get_display_image()
        yield im, f"生成中断！\n使用的Prompt:\n{combined_prompt}"

def generate_original_rotation(prompt1, prompt2, prompt3, prompt4, prompt5='', prompt6='', negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4, model_name="CompVis/stable-diffusion-v1-4"):
    combined_prompt = f"{prompt1}\n{prompt2}\n{prompt3}\n{prompt4}\n{prompt5}\n{prompt6}"
    print('prompt a: ', prompt1)
    print('prompt b: ', prompt2)
    print('prompt c: ', prompt3)
    print('prompt d: ', prompt4)
    print()
    print('prompt for base 1 (opt.): ', prompt5)
    print('prompt for base 2 (opt.): ', prompt6)
    print('negative prompt: ', negative_prompt)
    gpu = torch.device('cuda')
    if 's' not in dir():
        s = sd.StableDiffusion(gpu, model_name)
    device = s.device
    label_w = NegativeLabel(prompt1,negative_prompt)
    label_x = NegativeLabel(prompt2,negative_prompt)
    label_y = NegativeLabel(prompt3,negative_prompt)
    label_z = NegativeLabel(prompt4,negative_prompt)

    label_p = NegativeLabel(prompt5,negative_prompt)
    label_q = NegativeLabel(prompt6,negative_prompt)
    learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, hidden_dim=256, num_features=128).to(s.device); SIZE=256

    bottom_image=learnable_image_maker()
    top_image=learnable_image_maker()
    brightness=3

    CLEAN_MODE = True # If it's False, we augment the images by randomly simulating how good a random printer might be when making the overlays...

    def simulate_overlay(bottom, top):
        if CLEAN_MODE:
            exp=1
            brightness=3
            black=0
        else:
            exp=rp.random_float(.5,1)
            brightness=rp.random_float(1,5)
            black=rp.random_float(0,.5)
            bottom=rp.blend(bottom,black,rp.random_float())
            top=rp.blend(top,black,rp.random_float())
        return (bottom**exp * top**exp * brightness).clamp(0,99).tanh()
    learnable_image_w=lambda: simulate_overlay(bottom_image(), top_image().rot90(k=0,dims=[1,2]))
    learnable_image_x=lambda: simulate_overlay(bottom_image(), top_image().rot90(k=1,dims=[1,2]))
    learnable_image_y=lambda: simulate_overlay(bottom_image(), top_image().rot90(k=2,dims=[1,2]))
    learnable_image_z=lambda: simulate_overlay(bottom_image(), top_image().rot90(k=3,dims=[1,2]))
    
    learnable_image_p=lambda: bottom_image()
    learnable_image_q=lambda: top_image()
    params=chain(
        bottom_image.parameters(),
        top_image.parameters(),
    )
    optim=torch.optim.SGD(params,lr=1e-4)
    nums=[0,1,2,3]
    weights=[1,1,1,1]
    labels=[label_w,label_x,label_y,label_z]
    learnable_images=[learnable_image_w,learnable_image_x,learnable_image_y,learnable_image_z]
    if prompt5 != '' and prompt6 != '':
        nums=[0,1,2,3,4,5]
        weights=[1,1,1,1,1.4,1.4]
        labels=[label_w,label_x,label_y,label_z,label_p,label_q]
        learnable_images=[learnable_image_w,learnable_image_x,learnable_image_y,learnable_image_z,learnable_image_p,learnable_image_q]

    weights=rp.as_numpy_array(weights)
    weights=weights/weights.sum()
    weights=weights*len(weights)

    def get_display_image(border_color=(255, 255, 255)):
        return rp.tiled_images(
            [
                *[rp.as_numpy_image(image()) for image in learnable_images],
                rp.as_numpy_image(bottom_image()),
                rp.as_numpy_image(top_image()),
            ],
            length=len(learnable_images),
            border_thickness=2,
            border_color=border_color
        )
    global generating_flag
    generating_flag = True

    try:
        for iter_num in range(num_iter+1):
            if not generating_flag:
                break
            preds = []
            for label, learnable_image, weight in rp.random_batch(list(zip(labels, learnable_images, weights)), batch_size=1):
                pred = s.train_step(
                    label.embedding,
                    learnable_image()[None],
                    noise_coef=0.1 * weight, guidance_scale=100,
                )
                preds += list(pred)
            with torch.no_grad():
                if not iter_num % (display_interval // 4):
                    im = get_display_image()
                    if not iter_num % display_interval:
                        progress = int(iter_num / num_iter * 100)
                        yield im, f"进度：{progress}%\n使用的Prompt:\n{combined_prompt}"
            optim.step()
            optim.zero_grad()
    except Exception as e:
        print(f"Error occurred: {e}")
        yield None, f"生成失败: {str(e)}"
    except not generating_flag:
        print("\nInterrupted early.")
        im = get_display_image()
        yield im, f"生成中断！\n使用的Prompt:\n{combined_prompt}"

def generate_8_rotation(prompt_w, prompt_x, prompt_y, prompt_z,prompt_a,prompt_b,prompt_c,prompt_d, negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4, model_name="CompVis/stable-diffusion-v1-4"):
    combined_prompt = f"{prompt_w}\n{prompt_x}\n{prompt_y}\n{prompt_z}\n{prompt_a}\n{prompt_b}\n{prompt_c}\n{prompt_d}"

    print('    prompt_w =', (prompt_w))
    print('    prompt_x =', (prompt_x))
    print('    prompt_y =', (prompt_y))
    print('    prompt_z =', (prompt_z))
    print('    prompt_a =', (prompt_a))
    print('    prompt_b =', (prompt_b))
    print('    prompt_c =', (prompt_c))
    print('    prompt_d =', (prompt_d))
    gpu = torch.device('cuda')
    if 's' not in dir():
        s = sd.StableDiffusion(gpu, model_name)
    device = s.device
    label_w = NegativeLabel(prompt_w,negative_prompt)
    label_x = NegativeLabel(prompt_x,negative_prompt)
    label_y = NegativeLabel(prompt_y,negative_prompt)
    label_z = NegativeLabel(prompt_z,negative_prompt)
    label_a = NegativeLabel(prompt_a,negative_prompt)
    label_b = NegativeLabel(prompt_b,negative_prompt)
    label_c = NegativeLabel(prompt_c,negative_prompt)
    label_d = NegativeLabel(prompt_d,negative_prompt)
    learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, hidden_dim=256, num_features=128).to(s.device); SIZE=256
    bottom_image=learnable_image_maker()
    top_image=learnable_image_maker()

    brightness=3
    CLEAN_MODE = True # If it's False, we augment the images by randomly simulating how good a random printer might be when making the overlays...
    def simulate_overlay(bottom, top):
        if CLEAN_MODE:
            exp=1
            brightness=3
            black=0
        else:
            exp=rp.random_float(.5,1)
            brightness=rp.random_float(1,5)
            black=rp.random_float(0,.5)
            bottom=rp.blend(bottom,black,rp.random_float())
            top=rp.blend(top,black,rp.random_float())
        return (bottom**exp * top**exp * brightness).clamp(0,99).tanh()
    def rotate_image_tensor(image_tensor, angle):
        """
        使用仿射变换旋转张量图片
        :param image_tensor: [C, H, W] 的张量图片
        :param angle: 旋转角度（度数，逆时针为正）
        :return: 旋转后的张量图片
        """
        device=image_tensor.device
        angle = torch.tensor(angle * 3.1415926535 / 180,device=device)  # 角度转弧度
        theta = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0]
        ],device=device)  # 2x3 仿射矩阵

        theta = theta.unsqueeze(0)  # 扩展为批量大小为 1 的 3D 张量
        grid = torch.nn.functional.affine_grid(theta, image_tensor.unsqueeze(0).size(), align_corners=False)
        rotated_tensor = torch.nn.functional.grid_sample(image_tensor.unsqueeze(0), grid, align_corners=False)
        return rotated_tensor.squeeze(0)
        
    learnable_image_w=lambda: simulate_overlay(bottom_image(), rotate_image_tensor(top_image(),45))
    learnable_image_x=lambda: simulate_overlay(bottom_image(), rotate_image_tensor(top_image(),90))
    learnable_image_y=lambda: simulate_overlay(bottom_image(), rotate_image_tensor(top_image(),135))
    learnable_image_z=lambda: simulate_overlay(bottom_image(), rotate_image_tensor(top_image(),180))
    learnable_image_a=lambda: simulate_overlay(bottom_image(), rotate_image_tensor(top_image(),225))
    learnable_image_b=lambda: simulate_overlay(bottom_image(), rotate_image_tensor(top_image(),270))
    learnable_image_c=lambda: simulate_overlay(bottom_image(), rotate_image_tensor(top_image(),315))
    learnable_image_d=lambda: simulate_overlay(bottom_image(), rotate_image_tensor(top_image(),360))

    params=chain(
        bottom_image.parameters(),
        top_image.parameters(),
    )
    optim=torch.optim.SGD(params,lr=learning_rate)
    nums=[0,1,2,3,4,5,6,7]
    labels=[label_w,label_x,label_y,label_z,label_a,label_b,label_c,label_d]
    learnable_images=[learnable_image_w,learnable_image_x,learnable_image_y,learnable_image_z,learnable_image_a,learnable_image_b,learnable_image_c,learnable_image_d]

    #The weight coefficients for each prompt. For example, if we have [0,1,2,1], then prompt_w will provide no influence and prompt_y will have 1/2 the total influence
    weights=[1,1,1,1,1,1,1,1]

    labels=[labels[i] for i in nums]
    learnable_images=[learnable_images[i] for i in nums]
    weights=[weights[i] for i in nums]

    weights=rp.as_numpy_array(weights)
    weights=weights/weights.sum()
    weights=weights*len(weights)

    def get_display_image(border_color=(255, 255, 255)):
        return rp.tiled_images(
            [
                *[rp.as_numpy_image(image()) for image in learnable_images],
                rp.as_numpy_image(bottom_image()),
                rp.as_numpy_image(top_image()),
            ],
            length=len(learnable_images),
            border_thickness=2,
            border_color=border_color
        )
    global generating_flag
    generating_flag = True

    try:
        for iter_num in range(num_iter+1):
            if not generating_flag:
                break
            preds = []
            for label, learnable_image, weight in rp.random_batch(list(zip(labels, learnable_images, weights)), batch_size=1):
                pred = s.train_step(
                    label.embedding,
                    learnable_image()[None],
                    noise_coef=0.1 * weight, guidance_scale=100,
                )
                preds += list(pred)
            with torch.no_grad():
                if not iter_num % (display_interval // 4):
                    im = get_display_image()
                    if not iter_num % display_interval:
                        progress = int(iter_num / num_iter * 100)
                        yield im, f"进度：{progress}%\n使用的Prompt:\n{combined_prompt}"
            optim.step()
            optim.zero_grad()
    except Exception as e:
        print(f"Error occurred: {e}")
        yield None, f"生成失败: {str(e)}"
    except not generating_flag:
        print("\nInterrupted early.")
        im = get_display_image()
        yield im, f"生成中断！\n使用的Prompt:\n{combined_prompt}"

def generate_QR_highteracc(qr_content, prompt_a, prompt_b, prompt_z, negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4, model_name="CompVis/stable-diffusion-v1-4"):
    combined_prompt = f"{prompt_a}\n{prompt_b}\n{prompt_z}"
    print('QR code content:', qr_content)
    print()
    print('    prompt_a =', (prompt_a))
    print('    prompt_b =', (prompt_b))
    print('    prompt_c =', (prompt_z))

    def generate_qr_code(content, size=256):
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(content)
        qr.make(fit=True)

        qr_image = qr.make_image(fill_color="black", back_color="white").convert("L")
        qr_image = qr_image.resize((size, size), Image.Resampling.NEAREST)
        return qr_image
    
    def fade_tensor_image(tensor_image, alpha=0.5, target_value=1.0):
        if tensor_image.dtype != torch.float32:
            tensor_image = tensor_image.float()
        
        if tensor_image.max() > 1.0:
            tensor_image = tensor_image / 255.0

        faded_image = alpha * tensor_image + (1 - alpha) * target_value
        return faded_image
    qrc = generate_qr_code(qr_content, size=256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    qrc = transform(qrc).to('cuda')
    qrc = fade_tensor_image(qrc, alpha=0.3, target_value=1.0)

    if 's' not in dir():
        model_name="CompVis/stable-diffusion-v1-4"
        gpu='cuda:0'
        s=sd.StableDiffusion(gpu,model_name)
    device=s.device
    label_a = NegativeLabel(prompt_a,negative_prompt)
    label_b = NegativeLabel(prompt_b,negative_prompt)
    label_z = NegativeLabel(prompt_z,negative_prompt)

    learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, hidden_dim=256, num_features=128).to(s.device); SIZE=256
    image_a=learnable_image_maker()
    image_b=learnable_image_maker()
    k = 1
    learnable_image_a=lambda: k * image_a()
    learnable_image_b=lambda: k * image_b()
    learnable_image_z=lambda: (qrc - (image_a()+image_b())/2)

    params = chain(
        image_a.parameters(),
        image_b.parameters(),
    )
    optim=torch.optim.SGD(params,lr=1e-4)
    labels=[label_a, label_b, label_z]
    learnable_images=[learnable_image_a,learnable_image_b,learnable_image_z]
    weights=[1,1,10]
    weights=rp.as_numpy_array(weights)
    weights=weights/weights.sum()
    weights=weights*len(weights)

    
    def get_display_image(border_color=(255, 255, 255)):
        return rp.tiled_images(
            [
                *[rp.as_numpy_image(image()) for image in learnable_images],
                rp.as_numpy_image((learnable_image_a()/2+learnable_image_b()/2+learnable_image_z())),
            ],
            length=len(learnable_images),
            border_thickness=2,
            border_color=border_color
        )
    global generating_flag
    generating_flag = True

    try:
        for iter_num in range(num_iter+1):
            if not generating_flag:
                break
            preds = []
            for label, learnable_image, weight in rp.random_batch(list(zip(labels, learnable_images, weights)), batch_size=1):
                pred = s.train_step(
                    label.embedding,
                    learnable_image()[None],
                    noise_coef=0.1 * weight, guidance_scale=100,
                )
                preds += list(pred)
            with torch.no_grad():
                if not iter_num % (display_interval // 4):
                    im = get_display_image()
                    if not iter_num % display_interval:
                        progress = int(iter_num / num_iter * 100)
                        yield im, f"进度：{progress}%\n使用的Prompt:\n{combined_prompt}"
            optim.step()
            optim.zero_grad()
    except Exception as e:
        print(f"Error occurred: {e}")
        yield None, f"生成失败: {str(e)}"
    except not generating_flag:
        print("\nInterrupted early.")
        im = get_display_image()
        yield im, f"生成中断！\n使用的Prompt:\n{combined_prompt}"

def generate_QR_higherquality(qr_content, prompt_a, prompt_b, prompt_z, negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4, model_name="CompVis/stable-diffusion-v1-4"):
    combined_prompt = f"{prompt_a}\n{prompt_b}\n{prompt_z}"
    print('QR code content:', qr_content)
    print()
    print('    prompt_a =', (prompt_a))
    print('    prompt_b =', (prompt_b))
    print('    prompt_c =', (prompt_z))

    device = "cuda"
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)

    image = pipeline(prompt_z, num_inference_steps=50).images[0]
    
    background_path = "source/pics/QRbackground.png"
    image.save(background_path)
    output_path = "source/pics"
    output_file = f"{output_path}/custom_qrcode.png"

    amzqr.run(
        words=qr_content,
        picture=background_path,
        level='H', 
        version = 5, 
        colorized=True,
        save_name="custom_qrcode.png",
        save_dir=output_path,
    )

    def read_and_convert_image_to_tensor(image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        return transform(image).to('cuda')
    
    def fade_tensor_image(tensor_image, alpha=0.5, target_value=1.0):
        if tensor_image.dtype != torch.float32:
            tensor_image = tensor_image.float()
        if tensor_image.max() > 1.0:
            tensor_image = tensor_image / 255.0
        faded_image = alpha * tensor_image + (1 - alpha) * target_value
        return faded_image
    
    qrc = read_and_convert_image_to_tensor(output_file)
    qrc = fade_tensor_image(qrc, alpha=0.5, target_value=1.0)

    if 's' not in dir():
        model_name="CompVis/stable-diffusion-v1-4"
        gpu='cuda:0'
        s=sd.StableDiffusion(gpu,model_name)
    device=s.device
    label_a = NegativeLabel(prompt_a,negative_prompt)
    label_b = NegativeLabel(prompt_b,negative_prompt)
    label_z = NegativeLabel(prompt_z,negative_prompt)

    learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, hidden_dim=256, num_features=128).to(s.device); SIZE=256
    image_a=learnable_image_maker()
    image_b=learnable_image_maker()

    k = 1
    learnable_image_a=lambda: k * image_a()
    learnable_image_b=lambda: k * image_b()
    learnable_image_z=lambda: (3 * qrc - image_a() - image_b())

    params = chain(
        image_a.parameters(),
        image_b.parameters(),
    )
    optim=torch.optim.SGD(params,lr=1e-4)
    labels=[label_a, label_b, label_z]
    learnable_images=[learnable_image_a,learnable_image_b,learnable_image_z]

    weights=[1,1,6]

    weights=rp.as_numpy_array(weights)
    weights=weights/weights.sum()
    weights=weights*len(weights)

    ims=[]
    def get_display_image():
        display_image = rp.tiled_images(
            [
                *[rp.as_numpy_image(image()) for image in learnable_images],
                rp.as_numpy_image((learnable_image_a()+learnable_image_b()+learnable_image_z())/3),
            ],
            length=len(learnable_images),
            border_thickness=0,
        )
        return np.clip(display_image, -1, 1).astype(np.float32)
    global generating_flag
    generating_flag = True

    try:
        for iter_num in range(num_iter+1):
            if not generating_flag:
                break
            preds = []
            for label, learnable_image, weight in rp.random_batch(list(zip(labels, learnable_images, weights)), batch_size=1):
                pred = s.train_step(
                    label.embedding,
                    learnable_image()[None],
                    noise_coef=0.1 * weight, guidance_scale=100,
                )
                preds += list(pred)
            with torch.no_grad():
                if not iter_num % (display_interval // 4):
                    im = get_display_image()
                    if not iter_num % display_interval:
                        progress = int(iter_num / num_iter * 100)
                        yield im, f"进度：{progress}%\n使用的Prompt:\n{combined_prompt}"
            optim.step()
            optim.zero_grad()
    except Exception as e:
        print(f"Error occurred: {e}")
        yield None, f"生成失败: {str(e)}"
    except not generating_flag:
        print("\nInterrupted early.")
        im = get_display_image()
        yield im, f"生成中断！\n使用的Prompt:\n{combined_prompt}"

def generate_RGB(prompt_a, prompt_b,prompt_c,prompt_d, negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4, model_name="CompVis/stable-diffusion-v1-4"):
    combined_prompt = f"{prompt_a}\n{prompt_b}\n{prompt_c}\n{prompt_d}"
    print('    prompt_a =', (prompt_a))
    print('    prompt_b =', (prompt_b))
    print('    prompt_c =', (prompt_c))
    print('    prompt_d =', (prompt_d))
    
    if 's' not in dir():
        gpu='cuda:0'
        s=sd.StableDiffusion(gpu,model_name)

    device=s.device
    label_a = NegativeLabel(prompt_a,negative_prompt)
    label_b = NegativeLabel(prompt_b,negative_prompt)
    label_c = NegativeLabel(prompt_c,negative_prompt)
    label_d = NegativeLabel(prompt_d,negative_prompt)
    learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, hidden_dim=256, num_features=128).to(s.device); SIZE=256 
    image=learnable_image_maker()
    #保留红色通道
    def retain_red_channel(image):
        img_tensor = image()
        red_channel = img_tensor[0:1, :, :]
        zeros = torch.zeros_like(red_channel)
        red_image = torch.cat([red_channel, zeros, zeros], dim=0)
        return red_image

    # 保留绿色通道
    def retain_green_channel(image):
        img_tensor = image()
        green_channel = img_tensor[1:2, :, :]
        zeros = torch.zeros_like(green_channel)
        green_image = torch.cat([zeros, green_channel, zeros], dim=0)
        return green_image
    # 保留蓝色通道
    def retain_blue_channel(image):
        img_tensor = image()
        blue_channel = img_tensor[2:3, :, :]
        zeros = torch.zeros_like(blue_channel)
        blue_image = torch.cat([zeros, zeros, blue_channel], dim=0)
        return blue_image
    learnable_image_a = lambda: image()
    learnable_image_b = lambda: retain_red_channel(image)
    learnable_image_c = lambda: retain_blue_channel(image)
    learnable_image_d = lambda: retain_green_channel(image)
    
    optim=torch.optim.SGD(image.parameters(),lr=1e-4)

    labels=[label_a,label_b,label_c,label_d]
    learnable_images=[learnable_image_a,learnable_image_b,learnable_image_c,learnable_image_d]

    #The weight coefficients for each prompt. For example, if we have [0,1], then only the upside-down mode will be optimized
    weights=[1,1,1,1]

    weights=rp.as_numpy_array(weights)
    weights=weights/weights.sum()
    weights=weights*len(weights)

    def get_display_image():
        display_image = rp.tiled_images(
            [
                rp.as_numpy_image(learnable_image_a()),
                rp.as_numpy_image(learnable_image_b()),
                rp.as_numpy_image(learnable_image_c()),
                rp.as_numpy_image(learnable_image_d()),
                rp.as_numpy_image(image())
            ],
            length=len(learnable_images),
            border_thickness=0,
        )
        return np.clip(display_image, -1, 1).astype(np.float32)
    global generating_flag
    generating_flag = True

    try:
        for iter_num in range(num_iter+1):
            if not generating_flag:
                break
            preds = []
            for label, learnable_image, weight in rp.random_batch(list(zip(labels, learnable_images, weights)), batch_size=1):
                pred = s.train_step(
                    label.embedding,
                    learnable_image()[None],
                    noise_coef=0.1 * weight, guidance_scale=100,
                )
                preds += list(pred)
            with torch.no_grad():
                if not iter_num % (display_interval // 4):
                    im = get_display_image()
                    if not iter_num % display_interval:
                        progress = int(iter_num / num_iter * 100)
                        yield im, f"进度：{progress}%\n使用的Prompt:\n{combined_prompt}"
            optim.step()
            optim.zero_grad()
    except Exception as e:
        print(f"Error occurred: {e}")
        yield None, f"生成失败: {str(e)}"
    except not generating_flag:
        print("\nInterrupted early.")
        im = get_display_image()
        yield im, f"生成中断！\n使用的Prompt:\n{combined_prompt}"

def generate_control_net(prompt, original_image, negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4):  
    controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet
    )
    image = pipe(
        prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=1.2, image=original_image
    ).images[0]
    output = make_image_grid([original_image, image], rows=1, cols=2)
    yield output, '100%'

def generate_hybrid(prompt_a, prompt_b, kernel_size, negative_prompt='', progress=0, num_iter=10000, display_interval=100, learning_rate=1e-4, model_name="CompVis/stable-diffusion-v1-4"):
    combined_prompt = f"{prompt_a}\n{prompt_b}"
    if 's' not in dir():
        gpu='cuda:0'
        s=sd.StableDiffusion(gpu,model_name)
    device=s.device
    label_a = NegativeLabel(prompt_a,negative_prompt)
    label_b = NegativeLabel(prompt_b,negative_prompt)
    learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, hidden_dim=256, num_features=128).to(s.device); SIZE=256
    image=learnable_image_maker()
    def gaussian_filter(image, kernel_size=7, sigma=3.0):
        """Applies a Gaussian filter to the input image."""
        device = image.device
        channels = image.size(0)
        x = torch.arange(kernel_size, device=device) - kernel_size // 2
        y = torch.arange(kernel_size, device=device) - kernel_size // 2
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(channels, 1, -1, -1)
        image = image.unsqueeze(0)
        filtered_image = torch.nn.functional.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)
        return filtered_image.squeeze(0)

    learnable_image_a = lambda: gaussian_filter(image(), kernel_size=kernel_size)
    learnable_image_b = lambda: image()

    optim=torch.optim.SGD(image.parameters(),lr=1e-4)

    labels=[label_a,label_b]
    learnable_images=[learnable_image_a,learnable_image_b]

    weights=[1,1]
    weights=rp.as_numpy_array(weights)
    weights=weights/weights.sum()
    weights=weights*len(weights)

    def get_display_image(border_color=(255, 255, 255)):
        display_image = rp.tiled_images(
            [
                rp.as_numpy_image(learnable_image_a()),
                rp.as_numpy_image(learnable_image_b()),
                rp.as_numpy_image(image())
            ],
            length=len(learnable_images),
            border_thickness=2,
            border_color=border_color
        )
        return np.clip(display_image, -1, 1).astype(np.float32)
    global generating_flag
    generating_flag = True

    try:
        for iter_num in range(num_iter+1):
            if not generating_flag:
                break
            preds = []
            for label, learnable_image, weight in rp.random_batch(list(zip(labels, learnable_images, weights)), batch_size=1):
                pred = s.train_step(
                    label.embedding,
                    learnable_image()[None],
                    noise_coef=0.1 * weight, guidance_scale=100,
                )
                preds += list(pred)
            with torch.no_grad():
                if not iter_num % (display_interval // 4):
                    im = get_display_image()
                    if not iter_num % display_interval:
                        progress = int(iter_num / num_iter * 100)
                        yield im, f"进度：{progress}%\n使用的Prompt:\n{combined_prompt}"
            optim.step()
            optim.zero_grad()
    except Exception as e:
        print(f"Error occurred: {e}")
        yield None, f"生成失败: {str(e)}"
    except not generating_flag:
        print("\nInterrupted early.")
        im = get_display_image()
        yield im, f"生成中断！\n使用的Prompt:\n{combined_prompt}"

# 页面切换逻辑
def switch_to_submenu(index):
    return (
        gr.update(visible=False),  # 主菜单隐藏
        gr.update(visible=(index == 1)),  # 子菜单1显示
        gr.update(visible=(index == 2)),  # 子菜单2显示
        gr.update(visible=(index == 3)),  # 子菜单3显示
    )

def switch_to_generate_page():
    return (
        gr.update(visible=False),  # 子菜单隐藏
        gr.update(visible=True),   # 生成页面显示
    )


def return_to_menu2():
    return (
        gr.update(visible=True),  # 主菜单显示
        gr.update(visible=False),  # 所有子菜单隐藏
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),  # 生成页面隐藏
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        "",  # 清空 negative prompt
        None,  # 清空结果图像
        "进度：0%",  # 重置进度条
    )
def return_to_menu4():
    return (
        gr.update(visible=True),  # 主菜单显示
        gr.update(visible=False),  # 所有子菜单隐藏
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),  # 生成页面隐藏
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        "",  # 清空 Prompt 3
        "",  # 清空 Prompt 4
        "",  # 清空 negative prompt
        None,  # 清空结果图像 
        "进度：0%",  # 重置进度条
    )
def return_to_menu6():
    return (
        gr.update(visible=True),  # 主菜单显示
        gr.update(visible=False),  # 所有子菜单隐藏
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),  # 生成页面隐藏
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        "",  # 清空 Prompt 3
        "",  # 清空 Prompt 4
        "",  # 清空 Prompt 5
        "",  # 清空 Prompt 6
        "",  # 清空 negative prompt
        None,  # 清空结果图像 
        "进度：0%",  # 重置进度条
    )
def return_to_menu8():
    return (
        gr.update(visible=True),  # 主菜单显示
        gr.update(visible=False),  # 所有子菜单隐藏
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),  # 生成页面隐藏
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        "",  # 清空 Prompt 3
        "",  # 清空 Prompt 4
        "",  # 清空 Prompt 5
        "",  # 清空 Prompt 6
        "",  # 清空 Prompt 7
        "",  # 清空 Prompt 8
        "",  # 清空 negative prompt
        None,  # 清空结果图像 
        "进度：0%",  # 重置进度条
    )
def return_to_menu_controlnet():
    return (
        gr.update(visible=True),  # 主菜单显示
        gr.update(visible=False),  # 所有子菜单隐藏
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),  # 生成页面隐藏
        "",  # 清空 Prompt 1
        None,  # 清空 Prompt image
        "",  # 清空 Prompt 3
        None,  # 清空结果图像 
        "进度：0%",  # 重置进度条
    )
def return_to_menu_hybrid():
    return (
        gr.update(visible=True),  # 主菜单显示
        gr.update(visible=False),  # 所有子菜单隐藏
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),  # 生成页面隐藏
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        11,
        "",  # 清空 negative
        None,  # 清空结果图像 
        "进度：0%",  # 重置进度条
    )
def reset_generation2():
    """
    单独终止生成并复原输入和输出。
    """
    global generating_flag
    generating_flag = False
    return (
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        "",
        None,  # 清空结果图像
        "进度：0%",  # 重置进度条
    )
def reset_generation4():
    """
    单独终止生成并复原输入和输出。
    """
    global generating_flag
    generating_flag = False
    return (
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        "",  # 清空 Prompt 3
        "",  # 清空 Prompt 4
        "",
        None,  # 清空结果图像
        "进度：0%",  # 重置进度条
    )
def reset_generation6():
    """
    单独终止生成并复原输入和输出。
    """
    global generating_flag
    generating_flag = False
    return (
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        "",  # 清空 Prompt 3
        "",  # 清空 Prompt 4
        "",  # 清空 Prompt 5
        "",  # 清空 Prompt 6
        "",
        None,  # 清空结果图像
        "进度：0%",  # 重置进度条
    )
def reset_generation8():
    """
    单独终止生成并复原输入和输出。
    """
    global generating_flag
    generating_flag = False
    return (
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        "",  # 清空 Prompt 3
        "",  # 清空 Prompt 4
        "",  # 清空 Prompt 5
        "",  # 清空 Prompt 6
        "",  # 清空 Prompt 7
        "",  # 清空 Prompt 8
        "",
        None,  # 清空结果图像
        "进度：0%",  # 重置进度条
    )
def reset_generation_controlnet():
    return (
        "",  # 清空 Prompt 1
        None,  # 清空 Prompt image
        "",  # 清空 Prompt 3
        None,  # 清空结果图像 
        "进度：0%",  # 重置进度条
    )
def reset_generation_hybrid():
    return (
        "",  # 清空 Prompt 1
        "",  # 清空 Prompt 2
        11,
        "",  # 清空 negative
        None,  # 清空结果图像 
        "进度：0%",  # 重置进度条
    )
# 加载示例 prompts
example_prompts = load_example_prompts()

# Gradio 界面布局
with gr.Blocks() as app:
    # 主菜单
    with gr.Row(visible=True) as main_menu:
        with gr.Column(scale=1):
            gr.Markdown("### CV final project - Diffusion illusion")
            gr.Markdown("欢迎使用我们的illusion扩展程序！")
        with gr.Column(scale=2):
            btn1 = gr.Button("1. 图像变换")
            btn2 = gr.Button("2. 隐藏信息")
            btn3 = gr.Button("3. 其他功能")

    # 子菜单1
    with gr.Row(visible=False) as submenu1:
        with gr.Column(scale=1):
            gr.Markdown("### 图像变换")
            btn1_1 = gr.Button("1. Generate Parker Puzzle (original)")
            btn1_2 = gr.Button("2. Generate Random Puzzle (Improvement 1.1)")
            btn1_3 = gr.Button("3. Generate Non-Rectangle Puzzle (Improvement 1.2) （待完善）")
            btn01_back = gr.Button("返回主菜单")
    # 子选项1-1生成页面
    with gr.Row(visible=False) as generate_page_option1:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/puzzle_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            prompt_input1_option1 = gr.Textbox(label="输入 Prompt 1", placeholder="请输入第一个 prompt")
            prompt_input2_option1 = gr.Textbox(label="输入 Prompt 2", placeholder="请输入第二个 prompt")
            negative_prompt_option1 = gr.Textbox(label="输入 Negative Prompt", placeholder="请输入negative prompt (optional)")
            btn_generate_option1 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_option1 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_option1 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_option1 = gr.Button("终止生成（单击暂停 双击取消）")
            btn1_back = gr.Button("返回主菜单")
    # 子选项1-2生成页面
    with gr.Row(visible=False) as generate_page_option2:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/random_puzzle_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            prompt_input1_option2 = gr.Textbox(label="输入 Prompt 1", placeholder="请输入第一个 prompt")
            prompt_input2_option2 = gr.Textbox(label="输入 Prompt 2", placeholder="请输入第二个 prompt")
            negative_prompt_option2 = gr.Textbox(label="输入 Negative Prompt", placeholder="请输入negative prompt (optional)")
            btn_generate_option2 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_option2 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_option2 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_option2 = gr.Button("终止生成（单击暂停 双击取消）")
            btn2_back = gr.Button("返回主菜单")


    # 子菜单2
    with gr.Row(visible=False) as submenu2:
        with gr.Column(scale=1):
            gr.Markdown("### 隐藏信息")
            btn2_1 = gr.Button("1. Rotation Overlay (original & improvement 3)")
            btn2_2 = gr.Button("2. Kaleidoscope Effect -- Rotating 45 Degree (Improvement 4)")
            btn2_3 = gr.Button("3. Hiding QRcode -- More Secret + Higher Scanning Accuracy (Improvement 7)")
            btn2_4 = gr.Button("4. Hiding QRcode -- Higher Image Quality (Improvement 7)")
            btn2_5 = gr.Button("5. Behind RGB Image (Improvement 6)")
            btn02_back = gr.Button("返回主菜单")
    # 子菜单2-1生成页面
    with gr.Row(visible=False) as generate_page_2_1:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/semantic_base_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            prompt_input_2_1_1 = gr.Textbox(label="输入 Prompt 1", placeholder="请输入第一个 prompt")
            prompt_input_2_1_2 = gr.Textbox(label="输入 Prompt 2", placeholder="请输入第二个 prompt")
            prompt_input_2_1_3 = gr.Textbox(label="输入 Prompt 3", placeholder="请输入第三个 prompt")
            prompt_input_2_1_4 = gr.Textbox(label="输入 Prompt 4", placeholder="请输入第四个 prompt")
            prompt_input_2_1_5 = gr.Textbox(label="输入 Prompt 5 (optional.)", placeholder="请输入第五个 prompt")
            prompt_input_2_1_6 = gr.Textbox(label="输入 Prompt 6 (optional.)", placeholder="请输入第六个 prompt")
            negative_prompt_2_1 = gr.Textbox(label="输入 Negative Prompt", placeholder="请输入negative prompt (optional)")
            btn_generate_2_1 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_2_1 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_2_1 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_2_1 = gr.Button("终止生成（单击暂停 双击取消）")
            btn_back21 = gr.Button("返回主菜单")
    # 子菜单2-2生成页面
    with gr.Row(visible=False) as generate_page_2_2:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/Kaleidoscope_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            prompt_input_2_2_1 = gr.Textbox(label="输入 Prompt 1", placeholder="请输入第一个 prompt")
            prompt_input_2_2_2 = gr.Textbox(label="输入 Prompt 2", placeholder="请输入第二个 prompt")
            prompt_input_2_2_3 = gr.Textbox(label="输入 Prompt 3", placeholder="请输入第三个 prompt")
            prompt_input_2_2_4 = gr.Textbox(label="输入 Prompt 4", placeholder="请输入第四个 prompt")
            prompt_input_2_2_5 = gr.Textbox(label="输入 Prompt 5", placeholder="请输入第五个 prompt")
            prompt_input_2_2_6 = gr.Textbox(label="输入 Prompt 6", placeholder="请输入第六个 prompt")
            prompt_input_2_2_7 = gr.Textbox(label="输入 Prompt 7", placeholder="请输入第七个 prompt")
            prompt_input_2_2_8 = gr.Textbox(label="输入 Prompt 8", placeholder="请输入第八个 prompt")
            negative_prompt_2_2 = gr.Textbox(label="输入 Negative Prompt", placeholder="请输入negative prompt (optional)")
            btn_generate_2_2 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_2_2 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_2_2 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_2_2 = gr.Button("终止生成（单击暂停 双击取消）")
            btn_back22 = gr.Button("返回主菜单")
    # 子菜单2-3生成页面
    with gr.Row(visible=False) as generate_page_2_3:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/QRacc_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            QR_info1 = gr.Textbox(label="输入 QRcode 包含链接", placeholder="请输入包含的链接、文字等（长度对扫描效果有显著影响）")
            prompt_input_2_3_1 = gr.Textbox(label="输入 Prompt 1", placeholder="请输入第一个 prompt")
            prompt_input_2_3_2 = gr.Textbox(label="输入 Prompt 2", placeholder="请输入第二个 prompt")
            prompt_input_2_3_3 = gr.Textbox(label="输入 Prompt 3", placeholder="请输入第三个 prompt")
            btn_generate_2_3 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_2_3 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_2_3 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_2_3 = gr.Button("终止生成（单击暂停 双击取消）")
            btn_back23 = gr.Button("返回主菜单")
    # 子菜单2-4生成页面
    with gr.Row(visible=False) as generate_page_2_4:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/QRquality_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            QR_info2 = gr.Textbox(label="输入 QRcode 包含链接", placeholder="请输入包含的链接、文字等（长度对扫描效果有显著影响）")
            prompt_input_2_4_1 = gr.Textbox(label="输入 Prompt 1", placeholder="请输入第一个 prompt")
            prompt_input_2_4_2 = gr.Textbox(label="输入 Prompt 2", placeholder="请输入第二个 prompt")
            prompt_input_2_4_3 = gr.Textbox(label="输入 Prompt 3", placeholder="请输入第三个 prompt")
            btn_generate_2_4 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_2_4 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_2_4 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_2_4 = gr.Button("终止生成（单击暂停 双击取消）")
            btn_back24 = gr.Button("返回主菜单")
    # 子菜单2-5生成页面
    with gr.Row(visible=False) as generate_page_2_5:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/RGB_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            prompt_input_2_5_1 = gr.Textbox(label="输入 Prompt 1", placeholder="请输入第一个 prompt")
            prompt_input_2_5_2 = gr.Textbox(label="输入 Prompt 2", placeholder="请输入第二个 prompt")
            prompt_input_2_5_3 = gr.Textbox(label="输入 Prompt 3", placeholder="请输入第三个 prompt")
            prompt_input_2_5_4 = gr.Textbox(label="输入 Prompt 4", placeholder="请输入第四个 prompt")
            negative_prompt_2_5 = gr.Textbox(label="输入 Negative Prompt", placeholder="请输入negative prompt (optional)")
            btn_generate_2_5 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_2_5 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_2_5 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_2_5 = gr.Button("终止生成（单击暂停 双击取消）")
            btn_back25 = gr.Button("返回主菜单")

    # 子菜单3
    with gr.Row(visible=False) as submenu3:
        with gr.Column(scale=1):
            gr.Markdown("### 其他功能")
            btn3_1 = gr.Button("1. 歧义图像 With Controlnet")
            btn3_2 = gr.Button("2. 视距离幻觉 Hybrid")
            btn03_back = gr.Button("返回主菜单")
    # 子菜单3-1生成页面
    with gr.Row(visible=False) as generate_page_3_1:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/controlnet_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            prompt_input_3_1 = gr.Textbox(label="输入 Prompt 1", placeholder="town, medieval, landscapes, views,SFW, (masterpiece:1,2), best quality, masterpiece, highres, original, extremely detailed wallpaper, perfect lighting,(extremely detailed CG:1.2)")
            mask_image_3_1 = gr.Image(label="上传mask图像", type="pil")
            negative_prompt_3_1 = gr.Textbox(label="输入 Negative Prompt", placeholder="blurry, (bad anatomy:1.21), (bad proportions:1.3), extra limbs, (disfigured:1.3), (missing arms:1.3), (extra legs:1.3), (fused fingers:1.6), (too many fingers:1.6), (unclear eyes:1.3), lowers, bad hands, missing fingers, extra digit,bad hands, missing fingers, (((extra arms and legs)))")
            btn_generate_3_1 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_3_1 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_3_1 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_3_1 = gr.Button("终止生成（单击暂停 双击取消）")
            btn_back31 = gr.Button("返回主菜单")


    # 子菜单3-2生成页面
    with gr.Row(visible=False) as generate_page_3_2:
        with gr.Column(scale=1):
            gr.Image(value="source/pics/hybrid_demo.png", label="Demo 图像")
            with gr.Accordion("Example Prompts", open=True):
                gr.Textbox(value=example_prompts, label="Example Prompts", interactive=False, lines=10)
            prompt_input_3_2_1 = gr.Textbox(label="输入 Prompt 1", placeholder="请输入第一个 prompt")
            prompt_input_3_2_2 = gr.Textbox(label="输入 Prompt 2", placeholder="请输入第二个 prompt")
            kernel_size_3_2 = gr.Slider(
                label="选择高斯模糊核的大小", minimum=5, maximum=17, step=2, value=11
            )
            negative_prompt_3_2 = gr.Textbox(label="输入 Negative Prompt", placeholder="blurry, (bad anatomy:1.21), (bad proportions:1.3), extra limbs, (disfigured:1.3), (missing arms:1.3), (extra legs:1.3), (fused fingers:1.6), (too many fingers:1.6), (unclear eyes:1.3), lowers, bad hands, missing fingers, extra digit,bad hands, missing fingers, (((extra arms and legs)))")
            btn_generate_3_2 = gr.Button("开始生成")
        with gr.Column(scale=2):
            result_image_3_2 = gr.Image(label="生成的图像", interactive=False)
            progress_bar_3_2 = gr.Textbox(label="进度条", value="进度：0%", interactive=False)
            btn_stop_3_2 = gr.Button("终止生成（单击暂停 双击取消）")
            btn_back32 = gr.Button("返回主菜单")

        


    # 主菜单跳转子菜单
    btn1.click(switch_to_submenu, inputs=[gr.State(1)], outputs=[main_menu, submenu1, submenu2, submenu3])
    btn2.click(switch_to_submenu, inputs=[gr.State(2)], outputs=[main_menu, submenu1, submenu2, submenu3])
    btn3.click(switch_to_submenu, inputs=[gr.State(3)], outputs=[main_menu, submenu1, submenu2, submenu3])

    # 子菜单1跳转生成页面
    btn1_1.click(switch_to_generate_page, inputs=[], outputs=[submenu1, generate_page_option1])
    btn1_2.click(switch_to_generate_page, inputs=[], outputs=[submenu1, generate_page_option2])
    btn01_back.click(
        return_to_menu2,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_option1, prompt_input1_option1, prompt_input2_option1, negative_prompt_option1, result_image_option1, progress_bar_option1],
    )

    # 子菜单2跳转生成页面
    btn2_1.click(switch_to_generate_page, inputs=[], outputs=[submenu2, generate_page_2_1])
    btn2_2.click(switch_to_generate_page, inputs=[], outputs=[submenu2, generate_page_2_2])
    btn2_3.click(switch_to_generate_page, inputs=[], outputs=[submenu2, generate_page_2_3])
    btn2_4.click(switch_to_generate_page, inputs=[], outputs=[submenu2, generate_page_2_4])
    btn2_5.click(switch_to_generate_page, inputs=[], outputs=[submenu2, generate_page_2_5])
    btn02_back.click(
        return_to_menu4,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3],
    )

    # 子菜单3跳转生成页面
    btn3_1.click(switch_to_generate_page, inputs=[], outputs=[submenu3, generate_page_3_1])
    btn3_2.click(switch_to_generate_page, inputs=[], outputs=[submenu3, generate_page_3_2])
    btn03_back.click(
        return_to_menu2,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3],
    )

    # 1-1生成页面操作
    btn_generate_option1.click(
        generate_parker_puzzle,
        inputs=[prompt_input1_option1, prompt_input2_option1, negative_prompt_option1],
        outputs=[result_image_option1, progress_bar_option1]
    )
    btn_stop_option1.click(
        reset_generation2,
        inputs=[],
        outputs=[prompt_input1_option1, prompt_input2_option1, negative_prompt_option1, result_image_option1, progress_bar_option1]
    )
    btn1_back.click(
        return_to_menu2,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_option1, prompt_input1_option1, prompt_input2_option1, negative_prompt_option1, result_image_option1, progress_bar_option1]
    )
    
    # 1-2生成页面操作
    btn2_back.click(
        return_to_menu2,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_option2, prompt_input1_option2, prompt_input2_option2, negative_prompt_option2, result_image_option2, progress_bar_option2]
    )
    btn_generate_option2.click(
        generate_random_puzzle,
        inputs=[prompt_input1_option2, prompt_input2_option2, negative_prompt_option2],
        outputs=[result_image_option2, progress_bar_option2]
    )
    btn_stop_option2.click(
        reset_generation2,
        inputs=[],
        outputs=[prompt_input1_option2, prompt_input2_option2, negative_prompt_option2, result_image_option2, progress_bar_option2]
    )

    # 2-1生成页面操作
    btn_generate_2_1.click(
        generate_original_rotation,
        inputs=[prompt_input_2_1_1, prompt_input_2_1_2, prompt_input_2_1_3, prompt_input_2_1_4, prompt_input_2_1_5, prompt_input_2_1_6, negative_prompt_2_1],
        outputs=[result_image_2_1, progress_bar_2_1]
    )
    btn_back21.click(
        return_to_menu6,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_2_1, prompt_input_2_1_1, prompt_input_2_1_2, prompt_input_2_1_3, prompt_input_2_1_4, prompt_input_2_1_5, prompt_input_2_1_6, negative_prompt_2_1, result_image_2_1, progress_bar_2_1]
    )
    btn_stop_2_1.click(
        reset_generation6,
        inputs=[],
        outputs=[prompt_input_2_1_1, prompt_input_2_1_2, prompt_input_2_1_3, prompt_input_2_1_4, prompt_input_2_1_5, prompt_input_2_1_6, negative_prompt_2_1, result_image_2_1, progress_bar_2_1]
    )

    # 2-2生成页面操作
    btn_generate_2_2.click(
        generate_8_rotation,
        inputs=[prompt_input_2_2_1, prompt_input_2_2_2, prompt_input_2_2_3, prompt_input_2_2_4, prompt_input_2_2_5, prompt_input_2_2_6, prompt_input_2_2_7, prompt_input_2_2_8, negative_prompt_2_2],
        outputs=[result_image_2_2, progress_bar_2_2]
    )
    btn_back22.click(
        return_to_menu8,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_2_2, prompt_input_2_2_1, prompt_input_2_2_2, prompt_input_2_2_3, prompt_input_2_2_4, prompt_input_2_2_5, prompt_input_2_2_6, prompt_input_2_2_7, prompt_input_2_2_8, negative_prompt_2_2, result_image_2_2, progress_bar_2_2]
    )
    btn_stop_2_2.click(
        reset_generation8,
        inputs=[],
        outputs=[prompt_input_2_2_1, prompt_input_2_2_2, prompt_input_2_2_3, prompt_input_2_2_4, prompt_input_2_2_5, prompt_input_2_2_6, prompt_input_2_2_7, prompt_input_2_2_8, negative_prompt_2_2, result_image_2_2, progress_bar_2_2]
    )

    # 2-3生成页面操作
    btn_generate_2_3.click(
        generate_QR_highteracc,
        inputs=[QR_info1, prompt_input_2_3_1, prompt_input_2_3_2, prompt_input_2_3_3],
        outputs=[result_image_2_3, progress_bar_2_3]
    )
    btn_back23.click(
        return_to_menu2,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_2_3, prompt_input_2_3_1, prompt_input_2_3_2, prompt_input_2_3_3, result_image_2_3, progress_bar_2_3]
    )
    btn_stop_2_3.click(
        reset_generation2,
        inputs=[],
        outputs=[prompt_input_2_3_1, prompt_input_2_3_2, prompt_input_2_3_3, result_image_2_3, progress_bar_2_3]
    )

    # 2-4生成页面操作
    btn_generate_2_4.click(
        generate_QR_higherquality,
        inputs=[QR_info2, prompt_input_2_4_1, prompt_input_2_4_2, prompt_input_2_4_3],
        outputs=[result_image_2_4, progress_bar_2_4]
    )
    btn_back24.click(
        return_to_menu2,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_2_4, prompt_input_2_4_1, prompt_input_2_4_2, prompt_input_2_4_3, result_image_2_4, progress_bar_2_4]
    )
    btn_stop_2_4.click(
        reset_generation2,
        inputs=[],
        outputs=[prompt_input_2_4_1, prompt_input_2_4_2, prompt_input_2_4_3, result_image_2_4, progress_bar_2_4]
    )

    # 2-5生成页面操作
    btn_generate_2_5.click(
        generate_RGB,
        inputs=[prompt_input_2_5_1, prompt_input_2_5_2, prompt_input_2_5_3, prompt_input_2_5_4, negative_prompt_2_5],
        outputs=[result_image_2_5, progress_bar_2_5]
    )
    btn_back25.click(
        return_to_menu4,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_2_5, prompt_input_2_5_1, prompt_input_2_5_2, prompt_input_2_5_3, prompt_input_2_5_4, negative_prompt_2_5, result_image_2_5, progress_bar_2_5]
    )
    btn_stop_2_5.click(
        reset_generation4,
        inputs=[],
        outputs=[prompt_input_2_5_1, prompt_input_2_5_2, prompt_input_2_5_3, prompt_input_2_5_4, negative_prompt_2_5, result_image_2_5, progress_bar_2_5]
    )

    # 3-1生成页面操作
    btn_generate_3_1.click(
        generate_control_net,
        inputs=[prompt_input_3_1, mask_image_3_1, negative_prompt_3_1],
        outputs=[result_image_3_1, progress_bar_3_1]
    )
    btn_back31.click(
        return_to_menu_controlnet,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_3_1, prompt_input_3_1, mask_image_3_1, negative_prompt_3_1, result_image_3_1, progress_bar_3_1]
    )
    btn_stop_3_1.click(
        reset_generation_controlnet,
        inputs=[],
        outputs=[prompt_input_3_1, mask_image_3_1, negative_prompt_3_1, result_image_3_1, progress_bar_3_1]
    )

    # 3-2生成页面操作
    btn_generate_3_2.click(
        generate_hybrid,
        inputs=[prompt_input_3_2_1, prompt_input_3_2_2, kernel_size_3_2, negative_prompt_3_2],
        outputs=[result_image_3_2, progress_bar_3_2]
    )
    btn_back32.click(
        return_to_menu_hybrid,
        inputs=[],
        outputs=[main_menu, submenu1, submenu2, submenu3, generate_page_3_2, prompt_input_3_2_1, prompt_input_3_2_2, kernel_size_3_2, negative_prompt_3_2, result_image_3_2, progress_bar_3_2]
    )
    btn_stop_3_1.click(
        reset_generation_hybrid,
        inputs=[],
        outputs=[prompt_input_3_2_1, prompt_input_3_2_2, kernel_size_3_2, negative_prompt_3_2, result_image_3_2, progress_bar_3_2]
    )

# 启动应用
app.launch()