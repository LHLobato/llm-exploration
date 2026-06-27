from torchvision import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from itertools import combinations
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import os
import torch 


def _all_estimators(base: dict) -> list[tuple[str, object]]:
    return [(name, clf()) for name, clf in base.items()]

BASE = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",       
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    ),
    "KNN": lambda: KNeighborsClassifier(
        n_neighbors=11,
        metric="cosine",           
        algorithm="brute",         
        n_jobs=-1,
    ),
    "XGB": lambda: XGBClassifier(
        n_estimators=400,
        max_depth=4,               
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.4,      
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
    "LinearSVC": lambda: LinearSVC(
        C=0.1,                    
        max_iter=2000,
        random_state=42,
    ),
}

_voting_entries = {
    f"Voting_{'_'.join(names)}": (
        lambda names=names: VotingClassifier(
            estimators=[(n, clf()) for n, clf in BASE.items() if n in names],
            voting="hard",
            n_jobs=-1,
        )
    )
    for r in range(2, len(BASE) + 1)
    for names in combinations(BASE.keys(), r)
}

_stacking_entries = {
    f"Stacking_{'_'.join(names)}": (
        lambda names=names: StackingClassifier(
            estimators=[(n, clf()) for n, clf in BASE.items() if n in names],
            final_estimator=LinearSVC(C=0.1, max_iter=2000, random_state=42),
            passthrough=False,
            n_jobs=-1,
        )
    )
    for r in range(2, len(BASE) + 1)
    for names in combinations(BASE.keys(), r)
}

registry = {**BASE, **_voting_entries, **_stacking_entries}


class TextDataset(Dataset):
    """Wraps arrays of texts and integer labels for use with DataLoader."""

    def __init__(self, texts: np.ndarray, labels: np.ndarray):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx]), int(self.labels[idx])


BERT_MODELS = {"BERT-Base", "distilBERT", "DEBERTa", "ModernBERT"}
 
 
class BertFeatureExtractor:
    """
    Extrai o embedding [CLS] da última camada oculta de modelos encoder
    (BERT, distilBERT, DeBERTa, ModernBERT) e salva/carrega em disco no
    mesmo formato .npz que FeatureExtractor usa para visão.
 
    Parâmetros
    ----------
    device      : 'cuda' ou 'cpu'
    model_name  : chave legível (ex.: 'BERT-Base', 'DEBERTa')
    model_path  : caminho HuggingFace ou local para AutoModel/AutoTokenizer
    max_length  : comprimento máximo de tokenização (default 200, igual ao treino)
    """
 
    def __init__(
        self,
        device: str,
        model_name: str,
        model_path: str,
        max_length: int = 200,
    ):
        if model_name not in BERT_MODELS:
            raise ValueError(
                f"model_name '{model_name}' não é um modelo encoder suportado. "
                f"Escolha entre: {BERT_MODELS}"
            )
 
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
 
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
 
 
    def set_datasets(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """Cria DataLoaders a partir dos arrays já divididos pelo script de treino."""
        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )
        self.train_loader = DataLoader(
            TextDataset(X_train, y_train), shuffle=False, **loader_kwargs
        )
        self.val_loader = DataLoader(
            TextDataset(X_val, y_val), shuffle=False, **loader_kwargs
        )
        self.test_loader = DataLoader(
            TextDataset(X_test, y_test), shuffle=False, **loader_kwargs
        )

 
    def _extract_features(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        feats, labels = [], []
 
        for texts, batch_labels in tqdm(loader, desc=f"Extracting [{self.model_name}]"):
            enc = self.tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
 
            with torch.no_grad():
                out = self.model(**enc)

            cls_embedding = out.last_hidden_state[:, 0, :]  
 
            feats.append(cls_embedding.cpu().float())
            labels.append(batch_labels)
 
        return torch.cat(feats).numpy(), torch.cat(labels).numpy()
 
    def extract_and_save(self, save_dir: str, dataset_name: str):
        """Extrai features dos três splits e salva em disco."""
        os.makedirs(save_dir, exist_ok=True)
 
        splits = [
            ("train", self.train_loader),
            ("val", self.val_loader),
            ("test", self.test_loader),
        ]
        for split, loader in splits:
            if loader is None:
                raise RuntimeError(
                    "Loaders não configurados. Chame set_datasets() primeiro."
                )
            feat_path = os.path.join(
                save_dir,
                f"{split}-{self.model_name}-{dataset_name}.npz",
            )
            if os.path.exists(feat_path):
                print(f"[SKIP] {feat_path} já existe.")
                continue
            X, y = self._extract_features(loader)
            np.savez(feat_path, X=X, y=y)
            print(f"[SAVED] {feat_path}  shape={X.shape}")
 
    @staticmethod
    def load_features(
        save_dir: str, model_name: str, dataset_name: str
    ) -> tuple[tuple, tuple, tuple]:
        """
        Carrega features salvas para os três splits.
 
        Retorna
        -------
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
 
        def _load(split):
            path = os.path.join(
                save_dir, f"{split}-{model_name}-{dataset_name}.npz"
            )
            data = np.load(path)
            return data["X"], data["y"]
 
        return _load("train"), _load("val"), _load("test")
