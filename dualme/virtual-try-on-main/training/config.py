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