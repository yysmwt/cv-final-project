from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# 这里面没有别的路径，只有一个输出路径，就输出到当前文件夹
# 使用的control net和stable diffusion版本是确定的，一共是20G左右的参数，自动下载自动运行

#这是原始图，直接改成相应图片就行
original_image = Image.open('monalisa.jpg')

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

canny_image = original_image
# image = cv2.Canny(image, low_threshold, high_threshold)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = Image.fromarray(image)
# make_image_grid([original_image, canny_image], rows=1, cols=2)



controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", controlnet=controlnet
)
prompt = "town, medieval, landscapes, views,SFW, (masterpiece:1,2), best quality, masterpiece, highres, original, extremely detailed wallpaper, perfect lighting,(extremely detailed CG:1.2),"
negative_prompt = 'blurry, (bad anatomy:1.21), (bad proportions:1.3), extra limbs, (disfigured:1.3), (missing arms:1.3), (extra legs:1.3), (fused fingers:1.6), (too many fingers:1.6), (unclear eyes:1.3), lowers, bad hands, missing fingers, extra digit,bad hands, missing fingers, (((extra arms and legs))),'

image = pipe(
    prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=1.2, image=canny_image
).images[0]
output = make_image_grid([canny_image, image], rows=1, cols=2)
output.save('output_s10.png')
