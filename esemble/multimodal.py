import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import numpy as np
import timm
import os
from tqdm import tqdm
import joblib
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.backends.cudnn as cudnn
from pyts.image import GramianAngularField
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BitsAndBytesConfig


class TextInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


class ImagePathDataset(Dataset):
    """Dataset que carrega imagens do disco sob demanda, evitando OOM."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

class Domain_Ensemble:
    def __init__(self, cnn_path, tabular_path, llm_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vision_model = timm.create_model('convnext_nano', pretrained=False, num_classes=2)

        print(f"[LOAD] CNN...")
        try:
            checkpoint = torch.load(cnn_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict_to_load = checkpoint['model_state_dict']
            else:
                state_dict_to_load = checkpoint

            self.vision_model.load_state_dict(state_dict_to_load)

        except Exception as e:
             raise RuntimeError(f"Erro ao carregar o checkpoint da CNN: {e}")
        self.vision_model.to(self.device)
        self.vision_model.eval()

        print("[LOAD] LLM...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.llm_model = AutoModelForSequenceClassification.from_pretrained(
            llm_path,
            quantization_config=bnb_config,
            num_labels=2
        )
        self.llm_model.to(self.device)
        self.llm_model.eval()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)

        print("[LOAD] Tabular model...")
        self.tabular_model = joblib.load(tabular_path)

        self.clf = LogisticRegression(random_state=42, max_iter=1000)
        print("[LOAD] Individual Models Loaded...")

    def _resolve_image_loader(self, X_image, batch_size):
        if isinstance(X_image, (np.ndarray, torch.Tensor)):
            tensor = torch.tensor(X_image, dtype=torch.float32) if isinstance(X_image, np.ndarray) else X_image.float()
            return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False), None

        if isinstance(X_image, (str, Path)):
            root = Path(X_image)
            EXTENSIONS = {'.png', '.jpg', '.jpeg'}

            # se tiver subpastas (malicious/normal/benign), coleta por pasta ordenado
            subfolders = sorted([p for p in root.iterdir() if p.is_dir()])

            if subfolders:
                paths, labels = [], []
                # mapeia nome da pasta → label
                label_map = {'malicious': 1, 'normal': 0, 'benign': 0}
                for folder in subfolders:
                    folder_label = label_map.get(folder.name.lower(), -1)
                    folder_files = sorted([
                        p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in EXTENSIONS
                    ])
                    paths.extend(folder_files)
                    labels.extend([folder_label] * len(folder_files))
            else:
                # diretório plano, sem subpastas
                paths = sorted([p for p in root.iterdir() if p.suffix.lower() in EXTENSIONS])
                labels = None

            X_image = paths
            self._image_labels_from_dir = np.array(labels) if labels else None

        # lista de paths
        if isinstance(X_image, list) and len(X_image) > 0 and isinstance(X_image[0], (str, Path)):
            dataset = ImagePathDataset([Path(p) for p in X_image])
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=self.device.type == 'cuda'
            ), X_image  # retorna também os paths resolvidos

        raise TypeError(f"X_image inválido: {type(X_image)}")
    def _get_base_predictions(self, X_image, X_tabular, X_text_raw, batch_size=256):
        """Gera previsões dos modelos base."""

        image_loader, resolved_paths = self._resolve_image_loader(X_image, batch_size)
        from_paths = isinstance(X_image, (str, Path, list)) and not isinstance(X_image, (np.ndarray, torch.Tensor))

        all_features = []
        with torch.no_grad():
            for batch in tqdm(image_loader, desc="Vision Predict"):
                # TensorDataset retorna (tensor,); ImagePathDataset retorna tensor direto
                image_batch = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
                features_batch = self.vision_model(image_batch)
                all_features.append(features_batch.detach().cpu().numpy())

        preds_vision = np.concatenate(all_features, axis=0)

        print("[LLM] Predicting...")
        text_dataset = TextInferenceDataset(X_text_raw, self.llm_tokenizer)
        text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)
        all_preds_llm = []
        with torch.no_grad():
            for batch in tqdm(text_loader, desc="  llm predict"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    outputs = self.llm_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                all_preds_llm.append(probs)

        preds_llm = np.concatenate(all_preds_llm, axis=0)[:, 1].reshape(-1, 1)

        print("[TABULAR] Predicting...")
        preds_tabular = self.tabular_model.predict_proba(X_tabular)[:, 1].reshape(-1, 1)

        return preds_vision, preds_llm, preds_tabular

    def fit(self, X_image, X_val_tabular, X_val_text_raw, y_val):
        """
        Treina o meta-modelo usando previsões do conjunto de validação.

        X_image pode ser:
            - np.ndarray / Tensor com imagens já carregadas
            - list de paths para imagens
            - str / Path de diretório raiz
        """
        print("[TRAIN] Collecting predictions from validation set...")
        p_vision, p_llm, p_tabular = self._get_base_predictions(X_image, X_val_tabular, X_val_text_raw)

        X_meta_train = np.hstack((p_vision, p_llm, p_tabular))

        print("[TRAIN] Training meta-model...")
        self.clf.fit(X_meta_train, y_val)
        print("[TRAIN] Meta-model training finished!")

    def predict_proba(self, X_image, X_tabular, X_text_raw):
        """
        Gera previsões finais do ensemble.

        X_image pode ser:
            - np.ndarray / Tensor com imagens já carregadas
            - list de paths para imagens
            - str / Path de diretório raiz
        """
        if not hasattr(self.clf, 'classes_'):
            raise RuntimeError("Meta-modelo não treinado. Chame .fit() primeiro.")

        p_vision, p_llm, p_tabular = self._get_base_predictions(X_image, X_tabular, X_text_raw)
        X_meta = np.hstack((p_vision, p_llm, p_tabular))

        return self.clf.predict_proba(X_meta)

    def predict(self, X_image, X_tabular, X_text_raw):
        probabilities = self.predict_proba(X_image, X_tabular, X_text_raw)
        return np.argmax(probabilities, axis=1)
