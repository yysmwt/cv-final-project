# 本文件主要记录我基于autodl主机实例中的环境复现‘parker puzzle’项目的过程
### Pipeline
# 配置环境 代理、包、细节修改
# 拷贝并运行代码
# 解释代码含义、梳理框架

### 1 配置环境
    # 1.1 下载所需要的文件
    # 下载网址（需翻墙）：https://github.com/RyannDaGreat/Diffusion-Illusions
    # 所需文件: source/. requirements.txt parker_puzzle_colab.ipynb(opt.)
    # 1.2 python环境
    # 终端输入：
    # pip install --upgrade -r /path/to/requirements.txt
    # pip install rp --upgrade
    # 具体所需版本记录于requirements.txt 应该是（未经检验）直接终端运行这两行就可以下载所有所需的包
    # 1.3 代理设置 proxy ！！重要！！
    # hugging face是最大的模型社区，其中有大量预训练的模型权重可供通过diffuers库下载
    # 然而不幸的是hf被中国大陆墙了，所以需要想办法绕过。
    # 可能的方案：
      # 1.3.1 使用镜像网站（路径名称和网址可能并不准确）
      # 查阅资料主要是通过在一个路径中含huggingface相关字段的constants.py文件中，
      # 将ENDPOINT=huggingface.co 修改为 hf-mirror.com的方法，但实践未成功
      # 1.3.2 使用clash代理
      # 没搞明白，而且以clash的几mb/s的最高下载速度，五六个g的权重有点慢。
      # 1.3.3 使用autodl提供的代理（成功）
      # 在autodl中租用的实例，平台提供了代理方案，具体介绍位于帮助文档 https://www.autodl.com/docs/network_turbo/
      # 请注意，使用终端运行方案会出现报错，目前尝试可行的方案如下
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
    # 太好了，你现在已经绕过了防火墙，成功访问hugging face了
    # 但我并没有找到什么好的测试环境方案
    # 建议直接打开source中的dino.py并运行之
    # 不过我不确定这是否需要完成hf登录，如果卡死或者报错，可以考虑先往后翻翻，完成hf登录，再来测试不迟。

    # 以上全部完成之后可能会面临rp库的一些问题，这是因为autodl提供的代理仅限github和huggingface，而rp库中从初始化就会有一些访问google.com的操作
    # 这会导致网络错误，报错节选如下：
    #
    #  if connected_to_internet():
    #  ^^^^^^^^^^^^^^^^^^^^^^^
    # File "/root/miniconda3/lib/python3.12/site-packages/rp/r.py", line 31175, in connected_to_internet
    # socket.create_connection(("www.google.com", 80))
    # File "/root/miniconda3/lib/python3.12/socket.py", line 837, in create_connection sock.connect(sa)
    # 
    # 解决方案：可以考虑在rp的源文件(一般名为r.py)查找connected_to_internet，将其中的测试网址google.com改成随便一个墙内的，比如我用bing.com

### 拷贝并运行代码

# prompt
prompt_a = "A awesome eagle with white head and brown body. Clear eyes and huge wings. Beautiful 3d picture. Flying above the bay."
prompt_b = "A young man holding a camera with white long lens. Taking photos of mountain in Switzerland. Anime-inspired."

# negative prompt
negative_prompt = 'blurry ugly'

print()
print('Negative prompt:',repr(negative_prompt))
print()
print('Chosen prompts:')
print('    prompt_a =', repr(prompt_a)) #This will be right-side up
print('    prompt_b =', repr(prompt_b)) #This will be upside-down

from rp import *
import numpy as np
import rp
import os

# 添加source文件夹所在路径，确保其中的文件被正确导入
import sys
sys.path.append('/root/autodl-tmp/cv-final-project')

import torch
import torch.nn as nn
import source.stable_diffusion as sd
from easydict import EasyDict
from source.learnable_textures import LearnableImageFourier
from source.stable_diffusion_labels import NegativeLabel
from itertools import chain
import time

### hugging face login
# 请在终端输入 huggingface-cli login
# 会产生登录交互，但你不用真的登录，你只需要按照指示去hf官网注册一个账号，然后创建一个新的token，并将token赋值给变量
# 我的这段代码可以直接登录hf，省去不少工作。
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

if 's' not in dir():
    #You can select the original Stable Diffusion 1.5 or some dreambooth of it
    # model_name="/root/autodl-tmp/cv-illusion/pretrained_model/stable-diffusion-v1.4"
    model_name="CompVis/stable-diffusion-v1-4"
    # model_name="/root/autodl-tmp/cv-illusion/pretrained_model/Arcane-Diffusion"
    
    gpu=torch.device('cuda')
    
    s=sd.StableDiffusion(gpu, model_name)

device=s.device

label_a = NegativeLabel(prompt_a,negative_prompt)
label_b = NegativeLabel(prompt_b,negative_prompt)


#Image Parametrization and Initialization (this section takes vram)

#Select Learnable Image Size (this has big VRAM显存 implications!):
#Note: We use implicit neural representations for better image quality
#They're previously used in our paper "TRITON: Neural Neural Textures make Sim2Real Consistent" (see tritonpaper.github.io)
# ... and that representation is based on Fourier Feature Networks (see bmild.github.io/fourfeat)
learnable_image_maker = lambda: LearnableImageFourier(height=256,width=256,num_features=256,hidden_dim=256,scale=10).to(s.device);SIZE=256

image=learnable_image_maker()


import torch
import torch.nn.functional as F

#This is the puzzle Matt used in his video!
uv_map_b = rp.load_image("/root/autodl-tmp/cv-final-project/improvement1-mask/parker_puzzle_uv_map.png")
uv_map_a = rp.get_identity_uv_map(*rp.get_image_dimensions(uv_map_b))


rp.display_image(uv_map_a)
rp.display_image(uv_map_b)

learnable_image_a = lambda: rp.apply_uv_map(image(), uv_map_a)
learnable_image_b = lambda: rp.apply_uv_map(image(), uv_map_b)

optim=torch.optim.SGD(image.parameters(),lr=1e-4)

labels=[label_a,label_b]
learnable_images=[learnable_image_a,learnable_image_b]

#The weight coefficients for each prompt. For example, if we have [0,1], then only the upside-down mode will be optimized
weights=[1,1]

weights=rp.as_numpy_array(weights)
weights=weights/weights.sum()
weights=weights*len(weights)




#For saving a timelapse
ims=[]

def get_display_image():
    return rp.tiled_images(
        [
            rp.as_numpy_image(learnable_image_a()),
            rp.as_numpy_image(learnable_image_b()),
        ],
        length=len(learnable_images),
        border_thickness=0,
    )




NUM_ITER=5000

#Set the minimum and maximum noise timesteps for the dream loss (aka score distillation loss)
s.max_step=MAX_STEP=990
s.min_step=MIN_STEP=10 

television = rp.JupyterDisplayChannel()
television.display()

display_eta=rp.eta(NUM_ITER, title='Status')

DISPLAY_INTERVAL = 200

print('Every %i iterations we display an image in the form [image_a, image_b], where'%DISPLAY_INTERVAL)
print('    image_a = (the right-side up image)')
print('    image_b = (image_a, but upside down)')
print()
print('Interrupt the kernel at any time to return the currently displayed image')
print('You can run this cell again to resume training later on')
print()
print('Please expect this to take quite a while to get good images (especially on the slower Colab GPU\'s)! The longer you wait the better they\'ll be')

try:
    for iter_num in range(NUM_ITER):
        display_eta(iter_num) #Print the remaining time

        preds=[]
        for label,learnable_image,weight in rp.random_batch(list(zip(labels,learnable_images,weights)), batch_size=1):
            pred=s.train_step(
                label.embedding,
                learnable_image()[None],

                #PRESETS (uncomment one):
                noise_coef=.1*weight,guidance_scale=100,#10
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

            if not iter_num%(DISPLAY_INTERVAL//4):
                im = get_display_image()
                ims.append(im)
                television.update(im)
                
                if not iter_num%DISPLAY_INTERVAL:
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
    folder="/output: %s"%name
    if rp.path_exists(folder):
        folder+='_%i'%time.time()
    rp.make_directory(folder)
    ims_names=['ims_%04i.png'%i for i in range(len(ims))]
    with rp.SetCurrentDirectoryTemporarily(folder):
        rp.save_images(ims,ims_names,show_progress=True)
    print()
    print('Saved timelapse to folder:',repr(folder))
    
save_run('-'.join([prompt_a,prompt_b])) #You can give it a good custom name if you want!