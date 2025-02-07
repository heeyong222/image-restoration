import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from skimage import img_as_ubyte

def split_image_into_patches(img, patch_size=512):
    """
    Split an image into non-overlapping patches of size patch_size x patch_size.
    Pads with black if the image dimensions are not a perfect multiple of patch_size.
    
    Args:
        img (PIL.Image): Input image
        patch_size (int): Size of each patch (default: 512)
    
    Returns:
        patches (list): List of image patches
        grid_size (tuple): Number of patches in (rows, cols)
    """
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
    """
    Merge patches back into a single image of the original size.

    Args:
        patches (list): List of PIL images (512x512)
        grid_size (tuple): (rows, cols) - Number of patches in each dimension
        original_size (tuple): Original (width, height)
        patch_size (int): Patch size (default: 512)

    Returns:
        PIL.Image: Reconstructed image
    """
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

def inference_and_save(model, input_path, save_path):
    """
    Perform inference on an image by splitting it into 512x512 patches,
    processing each patch through the model, and then merging them back.
    
    Args:
        model (torch.nn.Module): The trained model.
        input_path (str): Path to the input image.
        save_path (str): Path to save the output image.
    """
    # Load the image
    img = Image.open(input_path).convert('RGB')
    
    # Split image into patches
    patches, grid_size, original_size = split_image_into_patches(img, patch_size=512)

    # Convert patches to tensor and perform inference
    model.eval()
    processed_patches = []
    with torch.no_grad():
        for patch in patches:
            input_tensor = TF.to_tensor(patch).unsqueeze(0).cuda()  # Convert to tensor & add batch dim
            restored = model(input_tensor)
            restored = torch.clamp(restored, 0, 1)
            restored_np = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored_img = Image.fromarray(img_as_ubyte(restored_np[0]))
            processed_patches.append(restored_img)

    # Merge processed patches back into a single image
    restored_img = merge_patches_into_image(processed_patches, grid_size, original_size, patch_size=512)
    
    # Save the final restored image
    restored_img.save(save_path)