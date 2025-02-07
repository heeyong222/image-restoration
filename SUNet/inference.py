import argparse
import os
import torch
from torch import nn
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from skimage import img_as_ubyte
from model.SUNet import SUNet_model  

def split_image_into_patches(img, patch_size=512):
    w, h = img.size
    pad_w = (patch_size - w % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size

    # Pad the image with black (0) to make it divisible by patch_size
    img_padded = Image.new('RGB', (w + pad_w, h + pad_h), (0, 0, 0))
    img_padded.paste(img, (0, 0))

    padded_w, padded_h = img_padded.size
    patches = []
    
    # Split into non-overlapping patches
    for i in range(0, padded_h, patch_size):
        for j in range(0, padded_w, patch_size):
            patch = img_padded.crop((j, i, j + patch_size, i + patch_size))
            patches.append(patch)

    grid_size = (padded_h // patch_size, padded_w // patch_size)  # (rows, cols)
    return patches, grid_size, (w, h)

def merge_patches_into_image(patches, grid_size, original_size, patch_size=512):
    rows, cols = grid_size
    full_w, full_h = cols * patch_size, rows * patch_size  # Size of padded image
    merged_img = Image.new('RGB', (full_w, full_h))

    index = 0
    for i in range(rows):
        for j in range(cols):
            merged_img.paste(patches[index], (j * patch_size, i * patch_size))
            index += 1

    # Crop the image back to its original size
    merged_img = merged_img.crop((0, 0, original_size[0], original_size[1]))
    return merged_img

@torch.no_grad()
def inference_and_save(model, input_path, save_path, patch_size=512, device='cpu'):
    """
    Perform inference on an image by splitting it into patch_size x patch_size patches,
    processing each patch through the model, and then merging them back.
    """
    model.eval()

    # 1) Load the image
    img = Image.open(input_path).convert('RGB')
    
    # 2) Split image into patches
    patches, grid_size, original_size = split_image_into_patches(img, patch_size=patch_size)

    # 3) Convert patches to tensor and perform inference
    processed_patches = []
    for patch in patches:
        input_tensor = TF.to_tensor(patch).unsqueeze(0).to(device)  # [1,3,H,W], float in [0,1]
        restored = model(input_tensor)  # [1,3,H,W]
        restored = torch.clamp(restored, 0, 1)

        # -> Convert to PIL
        restored_np = restored.permute(0, 2, 3, 1).cpu().numpy()  # (1,H,W,3)
        restored_img = Image.fromarray(img_as_ubyte(restored_np[0]))  # skimage img_as_ubyte -> uint8
        processed_patches.append(restored_img)

    # 4) Merge processed patches
    merged_img = merge_patches_into_image(processed_patches, grid_size, original_size, patch_size=patch_size)
    
    # 5) Save the final restored image
    merged_img.save(save_path)
    print(f"[INFO] Inference complete. Result saved at: {save_path}")


def build_model(config):
    model = SUNet_model(config)  
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to config file (YAML or similar).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model .pth file.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image.")
    parser.add_argument("--output", type=str, default="result.png",
                        help="Path to save output image.")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Patch size for inference (default 512).")
    args = parser.parse_args()

    device = torch.device('cpu')

    config = None
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    model = build_model(config)
    model.to(device)

    
    print(f"[INFO] Loading checkpoint from {args.checkpoint}")
    # checkpoint 불러오
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_dict = model.state_dict()

    if 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint

    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if "attn_mask" in k:
            continue 
        
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            pass

    # 2) 모델 dict 업데이트 후 로드
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
        
        

    # 4) Run inference
    inference_and_save(model, args.input, args.output, patch_size=args.patch_size, device=device)


if __name__ == "__main__":
    main()