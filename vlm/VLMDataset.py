from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd 
import torch 


class VLMDataset(Dataset):
    def __init__(self, data_csv, tokenizer, model_name, max_length=256):
        self.csv = pd.read_csv(data_csv)
        self.image_paths = self.csv['image_path'].values
        self.prompts = self.csv['prompt'].values
        self.labels = self.csv['malicious'].values 
        self.tokenizer = tokenizer
        self.max_length = max_length

        if model_name == "MiniCNN":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                    std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(image)
        
        tokenized = self.tokenizer(
            self.prompts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",    
            return_tensors="pt"
        )
    
        return {
                "pixel_values": pixel_values,
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }