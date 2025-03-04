import yaml
import os
import torch
import glob
import shutil

from loguru import logger
from SUNet.model.SUNet import SUNet_model
from SUNet.utils.test_model_inference import *

class InferenceRunner:
    def __init__(self):
        logger.info("INIT || Inference Runner")
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def __call__(self, config_path):
        config = self._load_config(config_path)
        
        model_path = config['inference']['model']
        input_folder = config['inference']['inputs']
        gt_folder = config['inference']['gts']
        output_folder = config['inference']['outputs']
        
        os.makedirs(output_folder, exist_ok=True)
        
        model = SUNet_model(config)
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        image_paths = glob.glob(os.path.join(input_folder, '*.[pjPJ]*[npNP]*[gG]'))

        logger.debug(f"Found {len(image_paths)} images in {input_folder}.")
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)  
            name, ext = os.path.splitext(filename)
            
            output_inference_path = os.path.join(output_folder, f"{name}_output{ext}")
            
            output_original_path = os.path.join(output_folder, filename)
            
            logger.debug(f"Inferencing {filename} ...")
            
            inference_and_save(model, img_path, output_inference_path)
            
            shutil.copy2(img_path, output_original_path)
            gt_path = os.path.join(gt_folder, filename)
            if os.path.exists(gt_path):
                output_gt_path = os.path.join(output_folder, f"{name}_gt{ext}")
                shutil.copy2(gt_path, output_gt_path)
                logger.info(f"Copied ground truth: {gt_path} to {output_gt_path}")
            else:
                logger.warning(f"Ground truth for {filename} not found in {gt_folder}")
            logger.debug(f"Saved inference: {output_inference_path} and original: {output_original_path}")
        