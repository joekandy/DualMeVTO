import os
from torch.utils.data import Dataset
from PIL import Image

class VITONDataset(Dataset):
    def __init__(self, config):
        self.root = config.dataset_dir
        self.mode = config.dataset_mode
        self.pair_list = os.path.join(self.root, config.dataset_list)
        self.resolution = config.resolution
        with open(self.pair_list, 'r') as f:
            self.pairs = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        person_name, cloth_name = self.pairs[idx]
        person_path = os.path.join(self.root, 'image', person_name)
        cloth_path = os.path.join(self.root, 'cloth', cloth_name)
        person_img = Image.open(person_path).convert('RGB').resize(self.resolution)
        cloth_img = Image.open(cloth_path).convert('RGB').resize(self.resolution)
        # Qui puoi aggiungere eventuali trasformazioni o return personalizzati
        return {'person': person_img, 'cloth': cloth_img, 'person_name': person_name, 'cloth_name': cloth_name} 