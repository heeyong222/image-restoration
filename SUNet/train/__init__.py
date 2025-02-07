from SUNet.train.train_new import *
from SUNet.train.train_pretrained import *
from loguru import logger

class TrainRunner:
    def __init__(self, pretrained):
        self.pretrained = pretrained
        logger.info("INIT | Train runner")
    
    def __call__(self, config_path):
        if self.pretrained is True:
            TrainWithPretrainedModel()(config_path)
        else:
            TrainNewModel()(config_path)