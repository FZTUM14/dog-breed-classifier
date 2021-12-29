import os
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from PIL import ImageFile
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib
import seaborn as sns
import torch

# %matplotlib inline
from IPython.display import set_matplotlib_formats
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from lightning_module_and_datasets import (
    MilestonesFinetuning,
    TransferLearningModel,
    DogBreedDataModule,
)

set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
# Setting the seed
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


log = logging.getLogger(__name__)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MilestonesFinetuning, "finetuning")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.link_arguments("finetuning.train_bn", "model.train_bn")
        parser.set_defaults(
            {
                "trainer.max_epochs": 180,
                "trainer.enable_model_summary": False,
                "trainer.num_sanity_val_steps": 1,
            }
        )


MyLightningCLI(
    TransferLearningModel,
    DogBreedDataModule,
    seed_everything_default=1234,
    trainer_defaults={"gpus": 1},
)
