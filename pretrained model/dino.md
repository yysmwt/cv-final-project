# dino-vitb16
hugging face网址：https://huggingface.co/facebook/dino-vitb16/tree/main
# 所需文件
- config.json
- pytorch_model.bin
# 调用方式
- 原调用dino_model = vit_base_patch16_224_dino(pretrained=True)
- 替换为：
  - dino_model = vit_base_patch16_224_dino(pretrained=False)
  - state_dict = torch.load('path_to_bin', weights_only=True)
  - dino_model.load_state_dict(state_dict)
  - dino_model.eval()
 
# 警告
- 没有找到妥善的方式解决警告：UserWarning: Mapping deprecated model name vit_base_patch16_224_dino to current vit_base_patch16_224.dino.
  dino_model = vit_base_patch16_224_dino(pretrained=False) 
  
# 报错
- 替换前报错（无法连接hf）
raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
