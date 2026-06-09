from peft import PeftModel
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
from CNN import build_model
from datetime import datetime
from MLP import MultiLayerPerceptron

def tprint(*args, **kwargs):
    """Print com timestamp automático."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{timestamp} |", *args, **kwargs)

from huggingface_hub import login

login(token="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")


SYSTEM_INSTRUCTION = (
    "You are a cybersecurity classifier. "
    "Analyze the domain record below and classify it as benign or malicious."
)

def make_loader(ds: TensorDataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=64, shuffle=shuffle,
                      num_workers=8, pin_memory=True)

def prepare_dataset(df: pd.DataFrame, n_samples: int | None = None,
                    random_state: int = 42) -> TensorDataset:
    if n_samples is not None and n_samples < len(df):
        # Amostragem estratificada por classe para manter proporção malicioso/benigno
        df = (
            df.groupby('malicious', group_keys=False)
              .apply(lambda g: g.sample(
                  min(len(g), round(n_samples * len(g) / len(df))),
                  random_state=random_state))
        )
    X = torch.tensor(
        df.drop(columns=['malicious', 'name', 'prompt','label','image_path'], errors='ignore').values,
        dtype=torch.float32
    )
    y = torch.tensor(df['malicious'].values, dtype=torch.long)
    return TensorDataset(X, y)


class TextInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=160):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _format(self, text: str) -> str:
        messages = [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{text}"}]
        return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def __len__(self):
            return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
                self._format(self.texts[idx]),
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
            transforms.Resize((42, 42)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

class Domain_Ensemble:
    def __init__(self, cnn_path, tab_path, llm_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        arch = {"name": "mid_3b_b", "blocks": [(32, 3, True),  (64, 3, True),  (128, 3, True)]}
        fc_dims = [512]
        self.vision_model = build_model(arch=arch, fc_dims=fc_dims, dropout=0.5, use_bn=True)

        tprint(f"[LOAD] CNN...")
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

        tprint("[LOAD] LLM...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.llm_model = AutoModelForSequenceClassification.from_pretrained(
           "meta-llama/Llama-3.2-1B-Instruct"
        )

        # Aplica o adapter
        self.llm_model = PeftModel.from_pretrained(self.llm_model, llm_path)
        self.llm_model.eval()
        self.llm_model.to(self.device)
        self.llm_model.eval()


        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path , local_files_only=True)
        
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model.config.pad_token_id = self.llm_tokenizer.eos_token_id

        tprint(f"[LOAD] Tabular model...")
        runs = {
            'model_type':     'MLP',
            'architecture':   [42, 256, 128, 64, 2],
            'activation':     nn.ReLU,
            'regularization': 0.0,
            'model_kwargs':   {},
            }

        self.tab_model = MultiLayerPerceptron(
            runs['architecture'],
            runs['regularization'],
            runs['activation'],
        ).to(self.device)

        try:
            checkpoint = torch.load(tab_path, map_location=self.device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                state_dict_to_load = checkpoint['model_state_dict']
            else:
                state_dict_to_load = checkpoint

            self.tab_model.load_state_dict(state_dict_to_load)

        except Exception as e:
            raise RuntimeError(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | [ERROR]: {e}")

        self.clf = LogisticRegression(random_state=42, max_iter=1000)
        tprint(f"[LOAD] Individual Models Loaded...")

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
    def _get_base_predictions(self, X_image, X_tabular, X_text_raw, batch_size=32):
        """Gera previsões dos modelos base."""

        image_loader, resolved_paths = self._resolve_image_loader(X_image, batch_size)
        from_paths = isinstance(X_image, (str, Path, list)) and not isinstance(X_image, (np.ndarray, torch.Tensor))

        all_features = []
        with torch.no_grad():
            for batch in tqdm(image_loader, desc=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | [VISION Predict]"):
                # TensorDataset retorna (tensor,); ImagePathDataset retorna tensor direto
                image_batch = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
                features_batch = self.vision_model(image_batch)
                all_features.append(features_batch.detach().cpu().numpy())

        preds_vision = np.concatenate(all_features, axis=0)

        tprint("[LLM] Predicting...")
        text_dataset = TextInferenceDataset(X_text_raw, self.llm_tokenizer)
        text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)
        all_preds_llm = []
        with torch.no_grad():
            for batch in tqdm(text_loader, desc=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | [LLM Predict]"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    outputs = self.llm_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                all_preds_llm.append(probs)

        preds_llm = np.concatenate(all_preds_llm, axis=0)[:, 1].reshape(-1, 1)

        tprint("[TABULAR] Predicting...")
        feat_ds = prepare_dataset(
                X_tabular,  # mantém o label pra prepare_dataset
                len(X_tabular)
            )
        feat_loader   = make_loader(feat_ds)

        all_preds_tab = []
        with torch.no_grad():
            for batch in tqdm(feat_loader, desc=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | [TABULAR Predict]"):
                outputs = self.tab_model(batch[0].to(self.device))
                probs = torch.softmax(outputs, dim=-1).detach().cpu().numpy()
                all_preds_tab.append(probs)

        preds_tab = np.concatenate(all_preds_tab, axis=0)[:, 1].reshape(-1, 1)


        return preds_vision, preds_llm, preds_tab

    def fit(self, X_image, X_val_tabular, X_val_text_raw, y_val):
        """
        Treina o meta-modelo usando previsões do conjunto de validação.

        X_image pode ser:
            - np.ndarray / Tensor com imagens já carregadas
            - list de paths para imagens
            - str / Path de diretório raiz
        """
        tprint("[TRAIN] Collecting predictions from validation set...")
        p_vision, p_llm, p_tabular = self._get_base_predictions(X_image, X_val_tabular, X_val_text_raw)

        X_meta_train = np.hstack((p_vision, p_llm, p_tabular))

        tprint("[TRAIN] Training meta-model...")
        self.clf.fit(X_meta_train, y_val)
        tprint("[TRAIN] Meta-model training finished!")

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
