import os
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

import sys
sys.path.append('/root/autodl-tmp/cv-final-project')

import numpy as np
import rp
import torch
import torch.nn as nn
import source.stable_diffusion as sd
from easydict import EasyDict
from source.learnable_textures import LearnableImageFourier
from source.stable_diffusion_labels import NegativeLabel
from itertools import chain
import time

#ONLY GOOD PROMPTS HERE
example_prompts = rp.load_yaml_file('/root/autodl-tmp/cv-final-project/source/example_prompts.yaml')
print('Available example prompts:', ', '.join(example_prompts))

#These prompts are all strings - you can replace them with whatever you want! By default it lets you choose from example prompts
prompt_w, prompt_x, prompt_y, prompt_z,prompt_a,prompt_b,prompt_c,prompt_d = rp.gather(example_prompts, 'mushroom gandalf winter burger miku pikachu pyramids lipstick'.split())

negative_prompt = ''

print()
print('Negative prompt:',repr(negative_prompt))
print()
print('Chosen prompts:')
print('    prompt_w =', repr(prompt_w))
print('    prompt_x =', repr(prompt_x))
print('    prompt_y =', repr(prompt_y))
print('    prompt_z =', repr(prompt_z))
print('    prompt_a =', repr(prompt_a))
print('    prompt_b =', repr(prompt_b))
print('    prompt_c =', repr(prompt_c))
print('    prompt_d =', repr(prompt_d))
if 's' not in dir():
    model_name="CompVis/stable-diffusion-v1-4"
    gpu='cuda:0'
    s=sd.StableDiffusion(gpu,model_name)
device=s.device

label_w = NegativeLabel(prompt_w,negative_prompt)
label_x = NegativeLabel(prompt_x,negative_prompt)
label_y = NegativeLabel(prompt_y,negative_prompt)
label_z = NegativeLabel(prompt_z,negative_prompt)
label_a = NegativeLabel(prompt_a,negative_prompt)
label_b = NegativeLabel(prompt_b,negative_prompt)
label_c = NegativeLabel(prompt_c,negative_prompt)
label_d = NegativeLabel(prompt_d,negative_prompt)

#Image Parametrization and Initialization (this section takes vram)

#Select Learnable Image Size (this has big VRAM implications!):
#Note: We use implicit neural representations for better image quality
#They're previously used in our paper "TRITON: Neural Neural Textures make Sim2Real Consistent" (see tritonpaper.github.io)
# ... and that representation is based on Fourier Feature Networks (see bmild.github.io/fourfeat)
learnable_image_maker = lambda: LearnableImageFourier(height=256, width=256, hidden_dim=256, num_features=128).to(s.device); SIZE=256
# learnable_image_maker = lambda: LearnableImageFourier(height=512,width=512,num_features=256,hidden_dim=256,scale=20).to(s.device);SIZE=512

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
optim=torch.optim.SGD(params,lr=1e-4)
nums=[0,1,2,3,4,5,6,7]

#Uncommenting one of the lines will disable some of the prompts, in case you don't want to use all four for some reason (like the Summer/Winter example)
# nums=[0  ,2,3]
# nums=[    2  ]
# nums=[0,1,2]
# nums=[1]
# nums=[0,1]
# nums=[0,2]


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
#For saving a timelapse
ims=[]
def get_display_image():
    return rp.tiled_images(
        [
            *[rp.as_numpy_image(image()) for image in learnable_images],
            rp.as_numpy_image(bottom_image()),
            rp.as_numpy_image(top_image()),
        ],
        length=len(learnable_images),
        border_thickness=0,
    )
NUM_ITER=10000

#Set the minimum and maximum noise timesteps for the dream loss (aka score distillation loss)
s.max_step=MAX_STEP=990
s.min_step=MIN_STEP=10 

display_eta=rp.eta(NUM_ITER, title='Status: ')

DISPLAY_INTERVAL = 200

print('Every %i iterations we display an image in the form [[image_w, image_x, image_y, image_z], [bottom_image, top_image]] where'%DISPLAY_INTERVAL)
print('    image_w = bottom_image * top_image')
print('    image_x = bottom_image * top_image.rot90()')
print('    image_y = bottom_image * top_image.rot180()')
print('    image_z = bottom_image * top_image.rot270()')
print()
print('Interrupt the kernel at any time to return the currently displayed image')
print('You can run this cell again to resume training later on')
print()
print('Please expect this to take hours to get good images (especially on the slower Colab GPU\'s! The longer you wait the better they\'ll be')

try:
    for iter_num in range(NUM_ITER):
        display_eta(iter_num) #Print the remaining time

        preds=[]
        for label,learnable_image,weight in rp.random_batch(list(zip(labels,learnable_images,weights)), batch_size=1):
            pred=s.train_step(
                label.embedding,
                learnable_image()[None],

                #PRESETS (uncomment one):
                noise_coef=.1*weight,guidance_scale=60,#10
                # noise_coef=0,image_coef=-.01,guidance_scale=50,
                # noise_coef=0,image_coef=-.005,guidance_scale=50,
                # noise_coef=.1,image_coef=-.010,guidance_scale=50,
                # noise_coef=.1,image_coef=-.005,guidance_scale=50,
                # noise_coef=.1*weight, image_coef=-.005*weight, guidance_scale=50,
            )
            preds+=list(pred)

        with torch.no_grad():
            if iter_num and not iter_num%(DISPLAY_INTERVAL*50):
                #Wipe the slate every 50 displays so they don't get cut off
                from IPython.display import clear_output
                clear_output()

            if not iter_num%DISPLAY_INTERVAL:
                im = get_display_image()
                ims.append(im)
                rp.display_image(im)

        optim.step()
        optim.zero_grad()
except KeyboardInterrupt:
    print()
    print('Interrupted early at iteration %i'%iter_num)
    im = get_display_image()
    ims.append(im)
    rp.display_image(im)

def save_run(name):
    folder="/root/autodl-tmp/cv-final-project/output-Kaleidoscope effect: %s"%name
    if rp.path_exists(folder):
        folder+='_%i'%time.time()
    rp.make_directory(folder)
    ims_names=['ims_%04i.png'%i for i in range(len(ims))]
    with rp.SetCurrentDirectoryTemporarily(folder):
        rp.save_images(ims,ims_names,show_progress=True)
    print()
    print('Saved timelapse to folder:',repr(folder))
    
save_run('1') #You can give it a good custom name if you want!
