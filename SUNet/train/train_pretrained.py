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
from SUNet.preprocessor.data_loader import *




class TrainWithPretrainedModel:
    def __init__(self):
        logger.info("INIT || Train with pretrained model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count()
        logger.debug(f"GPU counts {self.gpu_count}")
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def load_encoder_weights(self, model, checkpoint_path):
        """
        사전 학습된 모델의 가중치 중 '인코더' 부분만 로드하고,
        디코더 부분 (layers_up, up, norm_up 등)은 제외한다.
        """
        checkpoint = torch.load(checkpoint_path)
        # 보통 checkpoint가 {'state_dict': ...} 형태이므로 우선 가져옴.
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_dict = model.state_dict()

        pretrained_encoder = {}

        for key, value in state_dict.items():
            # ----------------
            # 1) 디코더 관련 키에 해당하면 스킵
            #    (layers_up, norm_up, up, concat_back_dim, output 등)
            # ----------------
            if (
                "layers_up" in key
                or "norm_up" in key
                or key.startswith("swin_unet.up")
                or "concat_back_dim" in key
                or key.startswith("swin_unet.output")
            ):
                # 디코더 부분으로 간주 -> 넘어감(skip)
                continue
            
            # ----------------
            # 2) 그 외(인코더라고 판단되는) 키만 반영
            # ----------------
            if key in model_dict and value.size() == model_dict[key].size():
                pretrained_encoder[key] = value
            else:
                # shape mismatch나 모델에 없는 키는 무시
                pass
        
        # 업데이트: 기존 모델 state dict에 선택된 encoder 가중치 덮어쓰기
        model_dict.update(pretrained_encoder)
        model.load_state_dict(model_dict)
        print("Pretrained encoder weights loaded successfully.")
        
    def load_partial_state_dict(self, model, checkpoint_path):
        """
        Loads a checkpoint into the model while only keeping matching layers.
        Raises an error if there are missing or mismatched layers (except for certain keys like attn_mask).
        """
        checkpoint = torch.load(checkpoint_path)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items()
                        if k in model_dict and v.size() == model_dict[k].size()}

        # Identify problematic layers
        missing_layers = [k for k in model_dict if k not in pretrained_dict]
        # Ignore keys containing 'attn_mask'
        missing_layers = [k for k in missing_layers if "attn_mask" not in k]
        
        if missing_layers:
            raise RuntimeError(f"Missing layers (exist in model but not in checkpoint): {missing_layers}")
        
        unexpected_layers = [k for k in checkpoint['state_dict'] if k not in model_dict]
        if unexpected_layers:
            raise RuntimeError(f"Unexpected layers (exist in checkpoint but not in model): {unexpected_layers}")
        
        # size_mismatch_layers = [k for k in checkpoint['state_dict']
        #                         if k in model_dict and checkpoint['state_dict'][k].size() != model_dict[k].size()]
        # if size_mismatch_layers:
        #     raise RuntimeError(f"Size mismatch layers: {size_mismatch_layers}")

        size_mismatch_layers = [k for k in checkpoint['state_dict']
                                if k in model_dict and checkpoint['state_dict'][k].size() != model_dict[k].size() 
                                and "attn_mask" not in k]
        if size_mismatch_layers:
            raise RuntimeError(f"Size mismatch layers: {size_mismatch_layers}")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.debug("Pretrained model successfully loaded")
    
    def __call__(self, config_path):
        config = self._load_config(config_path=config_path)
        EPOCHS = config['train']['epochs']
        model = SUNet_model(config)
        pretrained_model_path = config['train']['pretrain_path']
        # self.load_encoder_weights(model, pretrained_model_path)
        self.load_partial_state_dict(model, pretrained_model_path)
        p_number = network_parameters(model)
        
        if self.gpu_count > 1:
            model = nn.DataParallel(model)
        model.cuda()
        
        
        current_time = datetime.now().strftime('%y%m%d_%H%M%S')
        model_dir = os.path.join(config['train']['checkpoint_dirs'], current_time)
        os.makedirs(model_dir)
        
        writer = SummaryWriter(os.path.join(model_dir, 'logs'))
        
        train_dir = config['data']['train_dir']
        val_dir = config['data']['val_dir']
        
        optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['initial_lr']), betas=(0.9, 0.999), eps=1e-8, weight_decay=float(config['train']['weight_decay']))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # patience도 줄임
        # Loss function
        L1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss(layers=['relu3_3'], weights=[0.1])
        
        lambda_l1 = 1.0
        lambda_perc = 0.3      # Perceptual loss weight
        lambda_ssim = 0.3      # SSIM loss weight
        
        train_loader = DataLoader(
            dataset=DataLoaderTrain(train_dir, config['data']['patch_size']), batch_size=config['train']['batch_size'], shuffle=True, num_workers=0, drop_last=False
        )
        val_loader = DataLoader(
            dataset=DataLoaderValidation(val_dir, config['data']['patch_size']), batch_size=1, shuffle=True, num_workers=0, drop_last=False
        )
        
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
        
        # Training loop
        best_psnr = 0
        best_ssim = 0
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        logger.info("START || Training =============================")
        for epoch in tqdm(range(1, EPOCHS + 1), desc='Total Progress'):
            model.train()
            epoch_loss = 0
            for i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)):
                optimizer.zero_grad()
                target = data[0].cuda()
                input_ = data[1].cuda()

                restored = model(input_)
                # loss = L1_loss(restored, target)
                loss_l1 = L1_loss(restored, target)
                loss_perc = perceptual_loss(restored, target)  # Compute perceptual loss (e.g., using VGG features)
                loss_ssim = 1 - ssim_loss(restored, target)  # Compute SSIM loss; higher SSIM means lower loss

                # Combine losses with weights
                total_loss = (lambda_l1 * loss_l1 +
                            lambda_perc * loss_perc +
                            lambda_ssim * loss_ssim)
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                # loss.backward()
                # optimizer.step()
                # epoch_loss += loss.item()
            
            # calculate loss average
            avg_loss = epoch_loss / len(train_loader)
            logger.debug(f"\nEpoch {epoch} Average Loss: {avg_loss:.4f}")
            train_losses.append(avg_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # test after all epochs
            checkpoint_path = os.path.join(test_dir, f'model_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            checkpoint = torch.load(checkpoint_path)  # Load checkpoint
            model.load_state_dict(checkpoint['state_dict'])  # Restore model weights
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
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                with torch.no_grad():
                    restored = model(input_)
                restored = torch.clamp(restored, 0, 1)
                
                # L1 
                loss_l1_val = L1_loss(restored, target)
                val_losses.append(loss_l1_val.item())
                psnr_val_rgb.append(torchPSNR(restored, target))
                ssim_val_rgb.append(torchSSIM(restored, target))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
            
            # validation L1 mean
            avg_val_l1 = np.mean(val_losses)
            writer.add_scalar('Loss/val_l1', avg_val_l1, epoch)

            val_psnrs.append(psnr_val_rgb)
            val_ssims.append(ssim_val_rgb)
            
            logger.debug(f"PSNR: {psnr_val_rgb:.4f} SSIM: {ssim_val_rgb:.4f}")
            
            writer.add_scalar('PSNR/val', psnr_val_rgb, epoch)
            writer.add_scalar('SSIM/val', ssim_val_rgb, epoch)
            
            # Save best models
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

            # early_stopping(avg_val_loss, model, epoch, model_dir)
            # if early_stopping.early_stop:
            #     logger.info("Early stopping triggered. Exiting training loop.")
            #     break
            
            scheduler.step(-psnr_val_rgb)
            
        
        writer.close()
        logger.info("END || Training =============================")
        logger.debug(f'Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}')

            
        
        