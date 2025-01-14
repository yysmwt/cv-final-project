import os
import sys
sys.path.append('/root/autodl-tmp/cv-final-project')
sys.path.append('/root/autodl-tmp/cv-final-project/QR/QRsource')

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

from torchvision import transforms
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
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
from amzqr import amzqr
import qrcode


example_prompts = rp.load_yaml_file('improvement-prompt/example_prompt.yaml')
print('Available example prompts:', ', '.join(example_prompts))

prompt_a, prompt_b, prompt_z = rp.gather(example_prompts, 'froggo lipstick gold_coins'.split())

negative_prompt = ''


model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to(device)

image = pipeline(prompt_z, num_inference_steps=50).images[0]

output_path_1 = "generated_image.png"
image.save(output_path_1)

qr_content = "ThankYou!"
background_path = output_path_1

output_path = "improvement7-hiding QR/QR-outputs"
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

qrcode = read_and_convert_image_to_tensor(output_file)

print(f"二维码生成完成，保存路径：{output_path}/custom_qrcode.png")

def fade_tensor_image(tensor_image, alpha=0.5, target_value=1.0):
    if tensor_image.dtype != torch.float32:
        tensor_image = tensor_image.float()
    
    if tensor_image.max() > 1.0:
        tensor_image = tensor_image / 255.0

    faded_image = alpha * tensor_image + (1 - alpha) * target_value
    return faded_image

qrcode = fade_tensor_image(qrcode, alpha=0.5, target_value=1.0)

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
learnable_image_z=lambda: (3 * qrcode - image_a() - image_b())

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
    return rp.tiled_images(
        [
            *[rp.as_numpy_image(image()) for image in learnable_images],
            rp.as_numpy_image((learnable_image_a()+learnable_image_b()+learnable_image_z())/3),
        ],
        length=len(learnable_images),
        border_thickness=0,
    )
NUM_ITER=10000

s.max_step=MAX_STEP=990
s.min_step=MIN_STEP=10 

display_eta=rp.eta(NUM_ITER, title='Status: ')

DISPLAY_INTERVAL = 200

print('Every %i iterations we display an image in the form [image_a, image_b, image_c, image_d, image_z] where'%DISPLAY_INTERVAL)

try:
    for iter_num in range(NUM_ITER):
        display_eta(iter_num)

        preds=[]
        for label,learnable_image,weight in rp.random_batch(list(zip(labels,learnable_images,weights)), batch_size=1):
            pred=s.train_step(
                label.embedding,
                learnable_image()[None],

                noise_coef=.1*weight,guidance_scale=60,
            )
            preds+=list(pred)

        with torch.no_grad():
            if iter_num and not iter_num%(DISPLAY_INTERVAL*50):
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
    folder="improvement7-hiding QR/QR-outputs/hidden-QR: %s"%name
    if rp.path_exists(folder):
        folder+='_%i'%time.time()
    rp.make_directory(folder)
    ims_names=['ims_%04i.png'%i for i in range(len(ims))]
    with rp.SetCurrentDirectoryTemporarily(folder):
        rp.save_images(ims,ims_names,show_progress=True)
    print()
    print('Saved timelapse to folder:',repr(folder))

save_run('1')