import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

import timm

from PIL import Image

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# import timm

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
# from pytorch_metric_learning import losses
from torch.optim import lr_scheduler



class LitResNet50(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = models.resnet50(pretrained=True)
        self.finetune_layer = nn.Linear(self.backbone.fc.out_features, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['min_lr'])
        
        lr_scheduler_lit = {
            'scheduler': scheduler,
            'name': 'CosineAnnealingLR'
        }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
#         print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
#         print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
#         print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
#         print("Validation step ends")

        return BCEloss, preds, y
        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
#         print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        
        # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        # Save latest model checkpoint
        print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}")
        filename = f'/kaggle/temp/resnet50-siim-covid19-{epoch}-{mAPloss:.4f}.ckpt'
        trainer.save_checkpoint(filename)
        
        # Save model as Model Artifact
        artifact = wandb.Artifact(name=f'resnet50-{epoch}-{mAPloss:.4f}', type='model')
        artifact.add_file(filename)
        run.log_artifact(artifact)
        
#         print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output
    
    
    
#######################################################################


class LitResNet18(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = models.resnet18(pretrained=True)
        self.finetune_layer = nn.Linear(self.backbone.fc.out_features, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['min_lr'])
        
        lr_scheduler_lit = {
            'scheduler': scheduler,
            'name': 'CosineAnnealingLR'
        }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
        # print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
        # print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Validation step ends")

        return BCEloss, preds, y
        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        
        # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        # Save latest model checkpoint
        print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}")
        filename = f'/content/SIIM_COVID19/models/resnet18-siim-covid19-{epoch}-{mAPloss:.4f}.ckpt'
        trainer.save_checkpoint(filename)
        
        # Save model as Model Artifact
        artifact = wandb.Artifact(name=f'resnet18-{epoch}-{mAPloss:.4f}', type='model')
        artifact.add_file(filename)
        run.log_artifact(artifact)
        
        # print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output
    
    
    

##############################################################



class LitResNet18_2(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = models.resnet18(pretrained=True)
        self.finetune_layer = nn.Linear(self.backbone.fc.out_features, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['min_lr'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=Config['factor'], patience=Config['patience'], verbose=True, eps=Config['eps'])
        
        lr_scheduler_lit = {
            'scheduler': scheduler,
            'monitor': 'val/mAP',
            'name': 'ReduceLROnPlateau'
        }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
        # print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
        # print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Validation step ends")

        return BCEloss, preds, y
        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        
        # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        # Save latest model checkpoint
        print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}")
        filename = f'/content/SIIM_COVID19/models/resnet18-siim-covid19-{epoch}-{mAPloss:.4f}.ckpt'
        trainer.save_checkpoint(filename)
        
        # Save model as Model Artifact
        artifact = wandb.Artifact(name=f'resnet18-{epoch}-{mAPloss:.4f}', type='model')
        artifact.add_file(filename)
        run.log_artifact(artifact)
        
        # print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output
    
    
    
############################################################



class LitEffNetB0(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, in_chans=3, num_classes=512)
        # self.backbone = timm.create_model("efficientnet_b3a", pretrained=True, in_chans=3)
        self.finetune_layer = nn.Linear(512, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['min_lr'])
        
        lr_scheduler_lit = {
            'scheduler': scheduler,
            'name': 'CosineAnnealingLR'
        }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
        # print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
        # print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Validation step ends")

        return BCEloss, preds, y
        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        
        # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        # Save latest model checkpoint
        print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}")
        filename = f'/content/SIIM_COVID19/models/effnetb3a-siim-covid19-{epoch}-{mAPloss:.4f}.ckpt'
        trainer.save_checkpoint(filename)
        
        # Save model as Model Artifact
        artifact = wandb.Artifact(name=f'effnetb3a-{epoch}-{mAPloss:.4f}', type='model')
        artifact.add_file(filename)
        run.log_artifact(artifact)
        
        # print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output

    

    
#######################################


class LitEffNetB2a(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = timm.create_model('efficientnet_b2a', pretrained=True, in_chans=3, num_classes=self.hparams.num_classes)
#         self.backbone = timm.create_model("efficientnet_b2a", pretrained=True, in_chans=3)
#         self.finetune_layer = nn.Linear(512, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_preds = []
        self.train_y = []
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['min_lr'])
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=Config['factor'], patience=Config['patience'], verbose=True, eps=Config['eps'])
        
        lr_scheduler_lit = {
            'scheduler': scheduler,
            'name': 'CosineAnnealingLR'
        }

#         lr_scheduler_lit = {
#             'scheduler': scheduler,
#             'monitor': 'val/mAP',
#             'name': 'ReduceLROnPlateau'
#         }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
        # print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)

        self.train_preds.extend(preds.cpu().detach().numpy())
        self.train_y.extend(y.cpu().detach().numpy())
        
        # print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
        # print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Validation step ends")

        return BCEloss, preds, y
    


    def training_epoch_end(self, training_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch    
        
        # Calculate mAP score
        mAPloss = average_precision_score(self.train_y, self.train_preds)
        # Create logs
        self.log('train/mAP', mAPloss)


        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        
        # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        # Save latest model checkpoint
        print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}")
        filename = f'/content/SIIM_COVID19/models/effnetv2m-siim-covid19-{epoch}-{mAPloss:.4f}.ckpt'
        trainer.save_checkpoint(filename)
        
        # Save model as Model Artifact
        artifact = wandb.Artifact(name=f'effnetv2m-{epoch}-{mAPloss:.4f}', type='model')
        artifact.add_file(filename)
        run.log_artifact(artifact)
        
        # print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output

    
    
#####################################################


class LitEffNetB3a(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = timm.create_model('efficientnet_b3a', pretrained=True, in_chans=3, num_classes=self.hparams.num_classes)
#         self.backbone = timm.create_model("efficientnet_b3a", pretrained=True, in_chans=3)
#         self.finetune_layer = nn.Linear(512, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_preds = []
        self.train_y = []
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['min_lr'])
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=Config['factor'], patience=Config['patience'], verbose=True, eps=Config['eps'])
        
        lr_scheduler_lit = {
            'scheduler': scheduler,
            'name': 'CosineAnnealingLR'
        }

#         lr_scheduler_lit = {
#             'scheduler': scheduler,
#             'monitor': 'val/mAP',
#             'name': 'ReduceLROnPlateau'
#         }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
        # print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)

        self.train_preds.extend(preds.cpu().detach().numpy())
        self.train_y.extend(y.cpu().detach().numpy())
        
        # print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
        # print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Validation step ends")

        return BCEloss, preds, y
    


    def training_epoch_end(self, training_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch    
        
        # Calculate mAP score
        mAPloss = average_precision_score(self.train_y, self.train_preds)
        # Create logs
        self.log('train/mAP', mAPloss)


        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        
        # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        # Save latest model checkpoint
        print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}")
        filename = f'/content/SIIM_COVID19/models/effnetv2m-siim-covid19-{epoch}-{mAPloss:.4f}.ckpt'
        trainer.save_checkpoint(filename)
        
        # Save model as Model Artifact
        artifact = wandb.Artifact(name=f'effnetv2m-{epoch}-{mAPloss:.4f}', type='model')
        artifact.add_file(filename)
        run.log_artifact(artifact)
        
        # print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output
    


    
##########################################################

class LitEffNetV2_M(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = timm.create_model('efficientnetv2_m', pretrained=True, in_chans=3, num_classes=self.hparams.num_classes)
        # self.backbone = timm.create_model("efficientnet_b3a", pretrained=True, in_chans=3)
        # self.finetune_layer = nn.Linear(512, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_preds = []
        self.train_y = []
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['eta_min'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=Config['factor'], patience=Config['patience'], verbose=True, eps=Config['eps'])
#         scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=Config['T_0'], 
#                                                             #  T_mult=Config['T_mult'], 
#                                                              eta_min=Config['eta_min'], verbose=True)
        
#         lr_scheduler_lit = {
#             'scheduler': scheduler,
#             'name': 'CosineAnnealingWarmRestarts'
#         }

        lr_scheduler_lit = {
            'scheduler': scheduler,
            'monitor': 'val/mAP',
            'name': 'ReduceLROnPlateau'
        }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
        # print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)

        self.train_preds.extend(preds.cpu().detach().numpy())
        self.train_y.extend(y.cpu().detach().numpy())
        
        # print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
        # print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Validation step ends")

        return BCEloss, preds, y
    


    def training_epoch_end(self, training_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch    
        
        # Calculate mAP score
        mAPloss = average_precision_score(self.train_y, self.train_preds)
        # Create logs
        self.log('train/mAP', mAPloss)


        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        
        # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        # Save latest model checkpoint
        print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}")
        filename = f'/content/SIIM_COVID19/models/effnetv2m-siim-covid19-{epoch}-{mAPloss:.4f}.ckpt'
        trainer.save_checkpoint(filename)
        
        # Save model as Model Artifact
        # artifact = wandb.Artifact(name=f'effnetv2m-{epoch}-{mAPloss:.4f}', type='model')
        # artifact.add_file(filename)
        # run.log_artifact(artifact)
        
        # print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output

    
    
###############################################################


class LitTFEffNetV2_M(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = timm.create_model(self.hparams.model_name, pretrained=True, in_chans=3, num_classes=self.hparams.num_classes)
        # self.backbone = timm.create_model("efficientnet_b3a", pretrained=True, in_chans=3)
        # self.finetune_layer = nn.Linear(512, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_preds = []
        self.train_y = []
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['eta_min'])
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=Config['factor'], patience=Config['patience'], verbose=True, eps=Config['eps'])
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=Config['T_0'], 
                                                            #  T_mult=Config['T_mult'], 
                                                             eta_min=Config['eta_min'], verbose=True)
        
        lr_scheduler_lit = {
            'scheduler': scheduler,
            'name': 'CosineAnnealingWarmRestarts'
        }

        # lr_scheduler_lit = {
        #     'scheduler': scheduler,
        #     'monitor': 'val/mAP',
        #     'name': 'ReduceLROnPlateau'
        # }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
        # print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)

        self.train_preds.extend(preds.cpu().detach().numpy())
        self.train_y.extend(y.cpu().detach().numpy())
        
        # print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
        # print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Validation step ends")

        return BCEloss, preds, y
    


    def training_epoch_end(self, training_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch    
        
        # Calculate mAP score
        mAPloss = average_precision_score(self.train_y, self.train_preds)
        # Create logs
        self.log('train/mAP', mAPloss)


        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        
        # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        # Save latest model checkpoint
        print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}")
        filename = f'/content/SIIM_COVID19/models/tf_effnetv2m-siim-covid19-{epoch}-{mAPloss:.4f}.ckpt'
        trainer.save_checkpoint(filename)
        
        # Save model as Model Artifact
        artifact = wandb.Artifact(name=f'tf_effnetv2m-{epoch}-{mAPloss:.4f}', type='model')
        artifact.add_file(filename)
        run.log_artifact(artifact)
        
        # print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output
    
    
######################################################


class LitEffNetV2_S_IN21(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay, fold):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = timm.create_model(self.hparams.model_name, pretrained=True, in_chans=3, num_classes=self.hparams.num_classes)
        # self.backbone = timm.create_model("efficientnet_b3a", pretrained=True, in_chans=3)
        # self.finetune_layer = nn.Linear(512, self.hparams.num_classes)
        
        # compute the accuracy -- no need to roll your own!
        self.train_preds = []
        self.train_y = []
        
        self.mAP_best = 0
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'], eta_min=Config['eta_min'])
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=Config['factor'], patience=Config['patience'], verbose=True, eps=Config['eps'])
#         scheduler = lr_scheduler.CyclicLR(optimizer, cycle_momentum=False, base_lr=0.01, max_lr=0.1)
        
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=Config['T_0'], 
        #                                                      T_mult=Config['T_mult'], 
        #                                                      eta_min=Config['eta_min'], verbose=True)
        
        lr_scheduler_lit = {
            'scheduler': scheduler,
            'name': 'CosineAnnealingLR'
        }

        # lr_scheduler_lit = {
        #     'scheduler': scheduler,
        #     'monitor': 'val/mAP',
        #     'name': 'ReduceLROnPlateau'
        # }

        return [optimizer], [lr_scheduler_lit]

    
    def forward(self, x, y=None):
        
        x = x.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)

        return preds

    
    def training_step(self, batch, batch_idx):
        
        # print("Training step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('train/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)

        self.train_preds.extend(preds.cpu().detach().numpy())
        self.train_y.extend(y.cpu().detach().numpy())
        
        # print("Training step ends")
        
        return BCEloss#, preds, y
    
        
    
    def validation_step(self, batch, batch_idx):
        
        # print("Validation step starts")
        
        x, y = batch
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x = x.permute(0, 3, 1, 2)

        preds = self.backbone(x)
        # preds = self.finetune_layer(features)
        
        BCEloss = criterion1(preds, y)
        # Create logs
        self.log('val/BCEloss', BCEloss)
        
        preds = nn.functional.log_softmax(preds)
        
        # print("Validation step ends")

        return BCEloss, preds, y
    


    def training_epoch_end(self, training_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch    
        
        # Calculate mAP score
        mAPloss = average_precision_score(self.train_y, self.train_preds)
        # Create logs
        self.log('train/mAP', mAPloss)
        self.log('train/mAP(2/3)', (2*mAPloss)/3)


        
    
    def validation_epoch_end(self, val_step_outputs):
        """Called when the validation epoch ends."""
        
        # print("Validation epoch end starts")
        
        epoch = self.trainer.current_epoch
        
        preds = []
        y_true = []
        for out in val_step_outputs:
            _, pred, y = out
            preds.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
        
        # Calculate mAP score
        mAPloss = average_precision_score(y_true, preds)
        # Create logs
        self.log('val/mAP', mAPloss)
        self.log('val/mAP(2/3)', (2*mAPloss)/3)
        
        # # Histogram of logits
        self.logger.experiment.log(
        {"val/logits": wandb.Histogram(preds),
         "global_step": self.global_step})
        
        
        if mAPloss > self.mAP_best:
            print(f"Best mAP (Illi): {mAPloss}")
            self.mAP_best = mAPloss 
        
        
            # Save latest model checkpoint
            print(f"Saving latest model checkpoint at epoch {epoch}; mAP score: {mAPloss:.4f}; mAP(2/3) score: {(2*mAPloss)/3:.4f}")
            filename = f'/workspace/SIIM_COVID19/models/effnetb6_ns-siim-covid19-{epoch}-{mAPloss:.4f}-{self.hparams.fold}.ckpt'
            trainer.save_checkpoint(filename)

            # # Save model as Model Artifact
            artifact = wandb.Artifact(name=f'effnetb6_ns-{epoch}-{mAPloss:.4f}-{self.hparams.fold}', type='model')
            artifact.add_file(filename)
            run.log_artifact(artifact)

#             ! rm {filename}
        
        # print("Validation epoch end ends")
        
            
    
    def on_save_checkpoint(self, checkpoint):
        pass
            
    
    def predict(self, batch, batch_idx):
        x = batch
        output = self(x, y=None)
        return output
    
    
    
#############################################################




