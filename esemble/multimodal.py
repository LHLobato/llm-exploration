import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import numpy as np
import timm
import os
from tqdm import tqdm
import joblib
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
            num_labels=2)

        self.llm_model.to(self.device)
        self.llm_model.eval()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)

        print("[LOAD] Tabular model...")
        self.tabular_model = joblib.load(tabular_path)

        self.clf = LogisticRegression(random_state=42, max_iter=1000)
        print("[LOAD] Individual Models Loaded...")

    def _get_base_predictions(self, X_image, X_tabular, X_text_raw, batch_size=256):
        """ Gera previsões dos 2 modelos base. """
        if isinstance(X_image, np.ndarray):
             image_tensor = torch.tensor(X_image, dtype=torch.float32)
        else:
             image_tensor = X_image.float()

        temp_dataset = TensorDataset(image_tensor)
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

        all_features = []
        with torch.no_grad():
            for batch in tqdm(temp_loader, desc="Vision Predict"):
                image_batch = batch[0].to(self.device)
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
                with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
                    outputs = self.llm_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                all_preds_llm.append(probs)

        preds_llm = np.concatenate(all_preds_llm, axis=0)[:, 1].reshape(-1, 1)

        print("[TABULAR]  Predicting...")
        preds_tabular = self.tabular_model.predict_proba(X_tabular)[:, 1].reshape(-1, 1)

        return preds_vision, preds_llm, preds_tabular

    def fit(self, X_image, X_val_tabular, X_val_text_raw, y_val):
        """ Treina o meta-modelo (self.clf) usando previsões do conjunto de validação. """
        print("[TRAIN] loading predictions from validation set...")
        p_vision, p_llm, p_tabular = self._get_base_predictions(X_image, X_val_tabular, X_val_text_raw)

        X_meta_train = np.hstack((p_vision, p_llm, p_tabular))

        print("[TRAIN] Initializing Meta-model training...")
        self.clf.fit(X_meta_train, y_val)
        print("[TRAIN] Meta-model training finished!")

    def predict_proba(self, X_image, X_tabular, X_text_raw):
        """ Gera previsões finais do ensemble. """
        if not hasattr(self.clf, 'classes_'):
             raise RuntimeError("Meta-modelo não treinado. Chame .fit() primeiro.")

        p_vision, p_llm, p_tabular = self._get_base_predictions(X_image, X_tabular, X_text_raw)
        X_meta = np.hstack((p_vision, p_llm, p_tabular))

        return self.clf.predict_proba(X_meta)

    def predict(self, X_image, X_tabular, X_text_raw):
        probabilities = self.predict_proba(X_image, X_tabular, X_text_raw)
        return np.argmax(probabilities, axis=1)
