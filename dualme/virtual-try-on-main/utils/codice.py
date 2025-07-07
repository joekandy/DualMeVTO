# file: config.py

import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DDPMScheduler, StableDiffusionInpaintPipeline
import logging
import os


class VTONConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    learning_rate = 1e-5
    num_epochs = 50
    resolution = (1024, 768)
    dataset_dir = "./data/viton-hd"
    dataset_mode = "train"
    dataset_list = "train_pairs.txt"
    workers = 8
    shuffle = True
    checkpoint_interval = 5
    log_path = "./training.log"


class VTONDataModule:
    def __init__(self, config):
        from viton_hd_dataset import VITONDataset
        from torch.utils.data import DataLoader

        self.dataset = VITONDataset(config)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.workers,
            drop_last=True,
        )


class DualMeVTONModel:
    def __init__(self, config):
        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
        ).to(config.device)

    def save_checkpoint(self, epoch, path="./checkpoints"):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(f"{path}/dualme_vton_epoch_{epoch}")


class VTONLoss:
    def __init__(self, config):
        self.mse_loss = nn.MSELoss()
        self.device = config.device

    def compute(self, pred, target, mask):
        return self.mse_loss(pred * mask, target * mask)


class VTONLogger:
    def __init__(self, config):
        self.logger = logging.getLogger("DualMeVTONTrainer")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(config.log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def info(self, message):
        self.logger.info(message)
        print(message)


class DualMeVTONTrainer:
    def __init__(self):
        self.config = VTONConfig()
        self.logger = VTONLogger(self.config)
        self.data_module = VTONDataModule(self.config)
        self.model_module = DualMeVTONModel(self.config)
        self.loss_module = VTONLoss(self.config)
        self.optimizer = optim.AdamW(
            self.model_module.model.unet.parameters(), lr=self.config.learning_rate
        )
        os.makedirs("./checkpoints", exist_ok=True)

    def train(self):
        model = self.model_module.model
        model.unet.train()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_loss = 0
            for batch in self.data_module.loader:
                self.optimizer.zero_grad()

                src_image = batch["person"].to(self.config.device)
                garment_image = batch["cloth"].to(self.config.device)
                mask = torch.ones_like(src_image[:, :1]).to(self.config.device)

                with torch.no_grad():
                    latents = model.vae.encode(src_image).latent_dist.sample()
                    latents = latents * model.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    model.scheduler.num_train_timesteps,
                    (self.config.batch_size,),
                    device=self.config.device,
                )
                noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = model.text_encoder("virtual try-on")[0]

                noise_pred = model.unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = self.loss_module.compute(noise_pred, noise, mask)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.data_module.loader)
            self.logger.info(
                f"Epoch [{epoch}/{self.config.num_epochs}], Average Loss: {avg_loss:.4f}"
            )

            if epoch % self.config.checkpoint_interval == 0:
                self.model_module.save_checkpoint(epoch)
                self.logger.info(f"Checkpoint saved at epoch {epoch}")

        self.logger.info("Training completed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run DualMe VTON Training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    args = parser.parse_args()

    if args.mode == "train":
        trainer = DualMeVTONTrainer()
        trainer.train()
    elif args.mode == "eval":
        print("Evaluation mode not implemented yet.")


if __name__ == "__main__":
    main()
