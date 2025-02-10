import click
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from loguru import logger
from SUNet.train import TrainRunner
from SUNet.inference import InferenceRunner



@click.group
def cli():
    pass

@cli.command()
@click.option("--pretrained", is_flag=True, help="Use pretrained model for training")
@click.option("--config-path", type=click.Path(exists=True, readable=True), required=False, help="Configuration file path")
def train(pretrained, config_path):
    logger.info(f"INIT || train with {config_path}")
    TrainRunner(pretrained)(config_path)
    
@cli.command()
@click.option("--config-path", type=click.Path(exists=True, readable=True), required=False, help="Configuration file path")
def inference(config_path):
    logger.info(f"INIT || inference {config_path}")
    InferenceRunner()(config_path)
    
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    current_dir = Path(__file__).resolve().parent
    logger.add(current_dir / "SUNet.log")
    cli()