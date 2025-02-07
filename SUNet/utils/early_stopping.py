import torch
import os

from loguru import logger

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): Validation metric이 개선되지 않는 에포크 수.
            verbose (bool): 개선 시 메시지를 출력할지 여부.
            delta (float): 최소 성능 개선폭.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_metric, model, epoch, model_dir):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch, model_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.debug(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, epoch, model_dir)
            self.counter = 0

    def save_checkpoint(self, model, epoch, model_dir):
        """
        Validation 성능이 개선되었을 때 모델 저장.
        """
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(model_dir, "model_earlystop.pth"))
        if self.verbose:
            logger.debug("Validation metric improved. Saving model")