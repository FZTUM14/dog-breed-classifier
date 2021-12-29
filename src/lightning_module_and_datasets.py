import os
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sklearn
from torch.utils.data import Dataset
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib
import seaborn as sns
import torch

# %matplotlib inline
from IPython.display import HTML, display, set_matplotlib_formats
from PIL import Image
import logging
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info

set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
# Setting the seed
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


class DogBreedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_dir="/raid/user-data/fzimmermann/code_projects/project_dog_classification/data/dog_images/test",
        transform=None,
        target_transform=None,
        limit_images=None,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.limit_images = limit_images

    def setup(self, stage: Optional[str] = None):
        # normalization needed for input to pre-trained nets
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.Resize(304),
                transforms.RandomCrop((269, 304)),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.test_valid_transforms = transforms.Compose(
            [
                transforms.Resize(304),
                transforms.CenterCrop((269, 304)),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.batch_size = 16
        self.data_transfer = {}
        self.data_transfer["train"] = DogDataset(
            "/raid/user-data/fzimmermann/code_projects/project_dog_classification/data/dog_images/train",
            transform=self.train_transforms,
        )
        self.data_transfer["valid"] = DogDataset(
            "/raid/user-data/fzimmermann/code_projects/project_dog_classification/data/dog_images/valid",
            transform=self.test_valid_transforms,
        )
        self.data_transfer["test"] = DogDataset(
            "/raid/user-data/fzimmermann/code_projects/project_dog_classification/data/dog_images/test",
            transform=self.test_valid_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_transfer["train"],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_transfer["valid"],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_transfer["test"],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )


class DogDataset(Dataset):
    def __init__(
        self,
        img_dir="/raid/user-data/fzimmermann/code_projects/project_dog_classification/data/dog_images/test",
        transform=None,
        target_transform=None,
        limit_images=None,
    ):
        labels = []
        images = []
        class_names = {}
        for folder in os.listdir(img_dir):
            label = (
                int(folder.split(".")[0]) - 1
            )  # folder names have labels 1-133, but labels have to start at 0 for pytorch
            class_names[label] = folder.split(".")[
                1
            ]  # class_names has for each lable index the corresponding class name
            image_paths = [
                os.path.join(img_dir, folder, img)
                for img in os.listdir(os.path.join(img_dir, folder))
            ]
            for img in image_paths:
                images.append(img)
                labels.append(label)
        self.image_dirs, self.img_labels = sklearn.utils.shuffle(
            images, labels, random_state=0
        )
        if limit_images:
            self.image_dirs = self.image_dirs[:limit_images]
            self.img_labels = self.img_labels[:limit_images]
        self.transform = transform
        self.target_transform = target_transform
        class_names = sorted(class_names.items())
        self.classes = [class_name[1] for class_name in class_names]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.image_dirs[idx]
        image = Image.open(img_path)
        # image is rgb already since PIL follows RGB color convention
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

log = logging.getLogger(__name__)


class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaing layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )




class TransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "mobilenet_v2",
        fc_input_size = 115200,
        train_bn: bool = False,
        milestones: tuple = (2, 4),
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.number_of_classes_in_train_set = 133
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.fc_input_size = fc_input_size

        self.__build_model()

        self.save_hyperparameters()

    def __build_model(self):
        """Define model layers & loss."""

        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        _fc_layers = [
            nn.Linear(self.fc_input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.Linear(300, self.number_of_classes_in_train_set),
        ]
        self.fc = nn.Sequential(*_fc_layers)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass.
        Returns logits.
        """

        x = self.feature_extractor(x)
        x = x.flatten(-3, -1)

        x = self.fc(x)

        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_logits = self.forward(x)

        train_loss = self.loss(y_logits, y_true)
        self.log("train_loss", train_loss, prog_bar=True)
        acc = (y_logits.argmax(dim=-1) == y_true).float().mean()
        self.log("train_acc", acc, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_logits = self.forward(x)

        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)
        acc = (y_logits.argmax(dim=-1) == y_true).float().mean()
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_logits = self.forward(x)

        # 2. Compute loss
        self.log("test_loss", self.loss(y_logits, y_true), prog_bar=True)
        acc = (y_logits.argmax(dim=-1) == y_true).float().mean()
        # 3. Compute accuracy:
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma
        )
        return [optimizer], [scheduler]


