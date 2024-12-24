# clip-vit-base-patch32
CLIP（Contrastive Language-Image Pre-training）模型是由OpenAI在2021年提出的一种多模态预训练神经网络。
该模型通过对比学习的方式，将图像和文本映射到同一个嵌入空间中，使得相关联的图像和文本在向量空间中彼此接近。
hugging face网址：https://huggingface.co/openai/clip-vit-base-patch32
# 所需文件：
！请将下述文件下载于用一个文件夹路径下 

config.json  

preprocessor_config.json  

pytorch_model.bin (主要权重文件) 

tokenizer.json  

vocab.json
# 调用方式
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") 
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 
请将上述文件路径替换为下载后的文件夹所在路径

# 报错
- 替换前报错（由于无法翻墙访问huggingface导致）
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like openai/clip-vit-base-patch32 is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
