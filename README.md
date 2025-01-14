# cv-final-project
这是计算机视觉课程大作业的项目代码。

# 项目要求
https://fancy-icebreaker-99b.notion.site/Creating-Visual-Cognitive-Illusions-158d8be58af48070b7bee41d7e6ea75d

# code base
本项目基于开源代码库Diffusion Illusions实现，github链接为 https://github.com/RyannDaGreat/Diffusion-Illusions.

# Step Tutorial 1: 环境搭建
- python环境: 所需要的库
    - pip install --upgrade -r requirements.txt 原代码中给出的包要求
    - pip install rp --upgrade
    - 其中还需要的库包括（可能重复）
        huggingface_hub easydict matplotlib PIL cv2 qrcode[pil] amzqr torchvision gradio pyyaml
- 代理设置: 代码中显式呈现
    - 我们在代码中显式地设置了autodl提供的代理，以供访问huggingface下载预训练模型，如果不是在autodl提供的服务器中运行可能需要相应调整
    - 登录huggingface，这一部分我们也在代码中基于huggingface_hub库实现，且我们在代码中添加了登陆的token，网络环境允许的情况下直接运行代码即可登录
    - 以上全部完成之后可能会面临rp库的一些问题，这是因为autodl提供的代理仅限github和huggingface，而rp库中从初始化就会有一些访问google.com的操作，这会导致网络错误，我们考虑在rp的源文件(一般名为r.py)查找connected_to_internet，将其中的测试网址google.com改成bing.com即可解决
- 参数设置：预训练模型下载
    - 所有的预训练模型下载均在代码中显式地使用diffuser库完成，如果可以正常访问hf即可进行，本模型使用了Stable Diffusion模型，以及Control Net模型。预计模型大小一共在20 ~ 30G之间，请提前预留好空间。

# Step Tutorial 2: GUI工具调用
为了更加简明快捷舒适清爽地生成对应的图片，我们的项目另外开发了一个GUI工具，可以方便地生成各种类型的幻觉图片。下面介绍如何部署并且运行。
1. 克隆项目：
'''
git clone https://github.com/yysmwt/cv-final-project.git
'''
2. 将工作目录设置为 /cv-final-project
3. 运行main.py。
4. 控制台中会出现一个本地链接，在浏览器中打开链接，即可通过我们开发的可视化工具进行幻觉图片生成。
5. 其中支持我们除了非矩形mask变换以外的绝大部分功能，只需点击菜单选择功能并根据给出的example prompts复制或者自己编写prompts输入即可开始生成对应的幻觉图片。生成过程可以停止和复原，生成的照片可以随时全屏或下载至本地。

# Step Tutorial 3: 直接执行代码
- 如果不希望使用gui，而是想要直接运行，即可打开improvements文件中的.py后缀文件直接运行，即可根据已经设置好的prompt生成对应幻觉图片。
- 注意，controlnet生成需要指导图片位于期望位置，请注意检查。


## Notice
再次提醒，本模型使用了Stable Diffusion模型，以及Control Net模型。预计模型大小一共在20 ~ 30G之间，请提前预留好空间。
