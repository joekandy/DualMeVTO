# file: config.py

import torch

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    learning_rate = 1e-5
    num_epochs = 50
    resolution = (1024, 768)
    lambda_leffa = 1e-3
    resolution_threshold = 1/32
    timestep_threshold = 500
    temperature_tau = 2.0
    dataset_dir = './data/viton-hd'
    dataset_mode = 'train'
    dataset_list = 'train_pairs.txt'
    semantic_nc = 13
    workers = 8
    shuffle = True
    pretrained_model_name_or_path = "stable-diffusion-inpainting"
    checkpoint_interval = 5
    log_path = './training.log'

# file: dataset_loader.py

from torch.utils.data import DataLoader
from viton_hd_dataset import VITONDataset
from config import Config

class DataModule:
    def __init__(self, config):
        self.dataset = VITONDataset(config)
        self.loader = DataLoader(
            self.dataset, batch_size=config.batch_size,
            shuffle=config.shuffle, num_workers=config.workers, drop_last=True
        )

# file: model.py

from leffa.model import LeffaModel
from config import Config
import torch

class ModelModule:
    def __init__(self, config):
        self.model = LeffaModel(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            new_in_channels=12,
            height=config.resolution[0],
            width=config.resolution[1],
            dtype="float16"
        ).to(config.device)

    def save_checkpoint(self, epoch, path='./checkpoints'):
        torch.save(self.model.state_dict(), f'{path}/model_epoch_{epoch}.pth')

# file: losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeffaLoss:
    def __init__(self, config):
        self.mse_loss = nn.MSELoss()
        self.resolution_threshold = config.resolution_threshold
        self.resolution = config.resolution
        self.device = config.device

    def compute(self, attention_maps, src_image, ref_image, mask):
        flow_fields = []
        for attn_map in attention_maps:
            if attn_map.shape[-1] / self.resolution[0] >= self.resolution_threshold:
                attn_map_avg = torch.mean(attn_map, dim=1)
                coords = torch.linspace(-1, 1, steps=attn_map_avg.size(-1), device=self.device)
                coord_map = torch.stack(torch.meshgrid(coords, coords), -1)
                flow_field = torch.einsum('bnhw,hwd->bnd', attn_map_avg, coord_map)
                flow_field_upsampled = F.interpolate(flow_field.permute(0, 3, 1, 2), size=self.resolution, mode='bilinear')
                warped_ref = F.grid_sample(ref_image, flow_field_upsampled.permute(0, 2, 3, 1))
                loss = self.mse_loss(warped_ref * mask, src_image * mask)
                flow_fields.append(loss)
        return torch.stack(flow_fields).mean()

# file: logger.py

import logging
from config import Config

class Logger:
    def __init__(self, config: Config):
        self.logger = logging.getLogger("LeffaTrainer")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(config.log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def info(self, message: str):
        self.logger.info(message)
        print(message)

# file: trainer.py

import torch
import torch.optim as optim
from diffusers import DDPMScheduler
from leffa.transform import LeffaTransform
from config import Config
from dataset_loader import DataModule
from model import ModelModule
from losses import LeffaLoss
from logger import Logger
import os

class Trainer:
    def __init__(self):
        self.config = Config()
        self.logger = Logger(self.config)
        self.data_module = DataModule(self.config)
        self.model_module = ModelModule(self.config)
        self.loss_module = LeffaLoss(self.config)
        self.optimizer = optim.AdamW(self.model_module.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = DDPMScheduler.from_pretrained(self.config.pretrained_model_name_or_path, subfolder="scheduler")
        self.transform = LeffaTransform(height=self.config.resolution[0], width=self.config.resolution[1]).to(self.config.device)
        os.makedirs('./checkpoints', exist_ok=True)

    def train(self):
        model = self.model_module.model
        model.train()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_loss = 0
            for batch in self.data_module.loader:
                self.optimizer.zero_grad()
                batch = self.transform(batch)

                src_image = batch['img'].to(self.config.device)
                ref_image = batch['cloth']['unpaired'].to(self.config.device)
                mask = batch['cloth_mask']['unpaired'].to(self.config.device)
                densepose = batch['pose'].to(self.config.device)

                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (self.config.batch_size,), device=self.config.device).long()
                noisy_src = self.scheduler.add_noise(model.vae_encode(src_image), torch.randn_like(src_image), timesteps)

                input_latent = torch.cat([noisy_src, mask, model.vae_encode(src_image * (1-mask)), densepose], dim=1)
                _, reference_features = model.unet_encoder(model.vae_encode(ref_image), timesteps)

                output = model.unet(input_latent, timesteps, reference_features=reference_features)[0]

                diffusion_loss = self.loss_module.mse_loss(output, noisy_src)

                if timesteps.max() < self.config.timestep_threshold:
                    attention_maps = model.unet.get_attention_maps()
                    leffa_loss = self.loss_module.compute(attention_maps, src_image, ref_image, mask)
                    loss = diffusion_loss + self.config.lambda_leffa * leffa_loss
                else:
                    loss = diffusion_loss

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.data_module.loader)
            self.logger.info(f'Epoch [{epoch}/{self.config.num_epochs}], Average Loss: {avg_loss:.4f}')

            if epoch % self.config.checkpoint_interval == 0:
                self.model_module.save_checkpoint(epoch)
                self.logger.info(f'Checkpoint saved at epoch {epoch}')

        self.logger.info("Training completed!")

# file: main.py

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Leffa Training or Evaluation")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode to run the script in')
    args = parser.parse_args()

    if args.mode == 'train':
        from trainer import Trainer
        trainer = Trainer()
        trainer.train()
    elif args.mode == 'eval':
        print("Evaluation mode not implemented yet.")

if __name__ == '__main__':
    main()
