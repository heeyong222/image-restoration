import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
import utils 
import shutil

from datetime import datetime
from loguru import logger
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from SUNet.model.SUNet import SUNet_model
from SUNet.utils.model_utils import *
from SUNet.utils.test_model_inference import *
from SUNet.utils.early_stopping import EarlyStopping
from SUNet.utils.image_utils import torchPSNR, torchSSIM
from SUNet.utils.loss_utils import PerceptualLoss, ssim_loss
from SUNet.utils.tqdm_log_handler import TqdmLoggingHandler
from SUNet.preprocessor.data_loader import *

class TrainNewModel:
    def __init__(self):
        logger.info("INIT || Train new model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count()
        logger.debug(f"GPU counts {self.gpu_count}")
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def __call__(self, config_path):
        # load config
        config = self._load_config(config_path=config_path)
        
        EPOCHS = config['train']['epochs']

        # model initialize
        model = SUNet_model(config)
        p_number = network_parameters(model)
        logger.debug(f"Total network parameters: {p_number}")
        
        if self.gpu_count > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        
        current_time = datetime.now().strftime('%y%m%d_%H%M%S')
        model_dir = os.path.join(config['train']['checkpoint_dirs'], current_time)
        os.makedirs(model_dir)
        
        writer = SummaryWriter(os.path.join(model_dir, 'logs'))
        
        train_dir = config['data']['train_dir']
        val_dir = config['data']['val_dir']
        
        train_loader = DataLoader(
            dataset=DataLoaderTrain(train_dir, config['data']['patch_size']), batch_size=config['train']['batch_size'], shuffle=True, num_workers=0, drop_last=False
        )
        val_loader = DataLoader(
            dataset=DataLoaderValidation(val_dir, config['data']['patch_size']), batch_size=1, shuffle=True, num_workers=0, drop_last=False
        )
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=float(config['train']['initial_lr']), 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            weight_decay=float(config['train']['weight_decay'])
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # set loss func
        L1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss(layers=['relu3_3'], weights=[0.1])

        lambda_l1 = 1.0
        lambda_perc = 1.0      # Perceptual loss weight
        lambda_ssim = 1.0    # SSIM loss weight
        
        
        test_dir = os.path.join(model_dir, 'samples')
        os.makedirs(test_dir, exist_ok=True)
        test_data_idx = 1
        test_images_path = []
        for test_image_path in config['data']['test_datas']:
            test_image_save_path = os.path.join(test_dir, f'test_{test_data_idx}.png')
            shutil.copy2(test_image_path, test_image_save_path)
            test_images_path.append(test_image_save_path)
            test_data_idx += 1

        train_losses = []
        val_psnrs = []
        val_ssims = []
        learning_rates = []

        best_psnr = 0
        best_ssim = 0
        early_stopping = EarlyStopping(patience=10, verbose=True)

        logger.info("START || Training =============================")
        for epoch in tqdm(range(1, EPOCHS + 1), desc='Total Progress', file=TqdmLoggingHandler()):
            model.train()
            epoch_loss = 0.0
            for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, file=TqdmLoggingHandler())):
                optimizer.zero_grad()
                target = data[0].to(self.device)
                input_ = data[1].to(self.device)

                restored = model(input_)

                loss_l1 = L1_loss(restored, target)
                loss_perc = perceptual_loss(restored, target)
                loss_ssim = 1 - ssim_loss(restored, target)

                total_loss = (lambda_l1 * loss_l1 +
                              lambda_perc * loss_perc +
                              lambda_ssim * loss_ssim)
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(train_loader)
            logger.debug(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
            train_losses.append(avg_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])

            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

            checkpoint_path = os.path.join(test_dir, f'model_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            logger.debug(f"START || Test image file inference epoch {epoch}")
            test_output_idx = 1
            for test_image_path in test_images_path:
                output_image_path = os.path.join(test_dir, f'test_output{test_output_idx}_{epoch}.png')
                inference_and_save(model, test_image_path, output_image_path)
                test_output_idx += 1
            logger.debug(f"END || Test image file inference epoch {epoch}")

            # validation
            model.eval()
            val_losses = []
            psnr_val_rgb = []
            ssim_val_rgb = []
            for ii, data_val in enumerate(tqdm(val_loader, desc='Validation', leave=False)):
                target = data_val[0].to(self.device)
                input_ = data_val[1].to(self.device)
                with torch.no_grad():
                    restored = model(input_)
                restored = torch.clamp(restored, 0, 1)

                # only l1 loss validation
                loss_l1_val = L1_loss(restored, target)
                val_losses.append(loss_l1_val.item())
                psnr_val_rgb.append(torchPSNR(restored, target))
                ssim_val_rgb.append(torchSSIM(restored, target))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

            avg_val_l1 = np.mean(val_losses)
            writer.add_scalar('Loss/val_l1', avg_val_l1, epoch)

            val_psnrs.append(psnr_val_rgb)
            val_ssims.append(ssim_val_rgb)

            logger.debug(f"Epoch {epoch}: PSNR: {psnr_val_rgb:.4f} SSIM: {ssim_val_rgb:.4f}")
            writer.add_scalar('PSNR/val', psnr_val_rgb, epoch)
            writer.add_scalar('SSIM/val', ssim_val_rgb, epoch)

            
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'psnr': best_psnr
                }, os.path.join(model_dir, "model_best_psnr.pth"))
                logger.debug(f"Saved best PSNR model. PSNR: {best_psnr:.4f}")

            if ssim_val_rgb > best_ssim:
                best_ssim = ssim_val_rgb
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ssim': best_ssim
                }, os.path.join(model_dir, "model_best_ssim.pth"))
                logger.debug(f"Saved best SSIM model. SSIM: {best_ssim:.4f}")

            # lr scheduler
            scheduler.step(-psnr_val_rgb)

            
            # early_stopping(avg_val_l1, model, epoch, model_dir)
            # if early_stopping.early_stop:
            #     logger.info("Early stopping triggered. Exiting training loop.")
            #     break

        writer.close()
        logger.info("END || Training =============================")
        logger.debug(f"Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}")
        
    