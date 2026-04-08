import os
import gc
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from pyts.image import GramianAngularField
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xgboost import XGBClassifier
import argparse
from MultiLayerPerceptron import MultiLayerPerceptron, train, test

parser = argparse.ArgumentParser()

parser.add_argument("--modelpath", type=str, help="Informe o diretório até a pasta do modelo")
parser.add_argument("--model", type=str, default="BERT-Base", choices=["BERT-Base", "distilBERT", "DEBERTa"],
                    help="Informe a arquitetura do modelo")
parser.add_argument("--classifier", type=str, default='RF', choices = ["RF", "XGB", "SVM", "LR", "MLP", "ENSEMBLE"], help="Machine Learning Model Classifier")
parser.add_argument("--ensemblepath", type=str, help="Caminho para o arquivo de estrutura do ensemble")
parser.add_argument("--ensemblemode", type=str, default="stacking", help="Modo de ensemble")
parser.add_argument("--fractais", action="store_true", help="Features Fractais")
parser.add_argument("--random_state", type=int, default=42)

args = parser.parse_args()

checkpoint_path = args.modelpath

model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

domain_names = pd.read_csv("../dns-feature-enrichment/csvs/dataset.csv", index_col=False)['name'].values
labels = pd.read_csv("../dns-feature-enrichment/csvs/dataset.csv", index_col=False)['malicious'].values

def extract_features_batch(texts, model, tokenizer, batch_size=32):
    all_features = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Extraindo Features DeBERTa (R)"):
        batch_texts = texts[i : i + batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            features = (sum_embeddings / sum_mask).cpu().numpy()
        all_features.append(features)
    return np.concatenate(all_features, axis=0)

state = 0 



X_text_train, X_text_temp, y_train, y_temp = train_test_split(
    domain_names, labels, test_size=0.3, random_state=state, stratify=labels
)
X_text_val, X_text_test, y_val, y_test = train_test_split(
    X_text_temp, y_temp, test_size=0.5, random_state=state, stratify=y_temp
)


X_train = extract_features_batch(X_text_train, model, tokenizer)
X_val = extract_features_batch(X_text_val, model, tokenizer)
X_test = extract_features_batch(X_text_test, model, tokenizer)

args = parser.parse_args()
if not args.modelpath:
    raise Exception("There must be a modelpath to load")

#if not args.image_folder:
#    raise Exception("There must be a imagefolder to load")
    
if not args.model:
    raise Exception("There must be a model architeture")

if not args.classifier:
    raise Exception("There must be a classifier architeture")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    'XGB': XGBClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=8, 
        subsample=0.6, colsample_bytree=0.6, gamma=0.2,
        tree_method='hist', eval_metric="auc", random_state=args.random_state, n_jobs=-1
    ),
    'RF': RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=args.random_state, n_jobs=-1
    ),

    'LR': SGDClassifier(
        loss='log_loss', penalty='l2', max_iter=2000, 
        class_weight='balanced', random_state=args.random_state, n_jobs=-1
    ),

    'SVM': SGDClassifier(
        loss='modified_huber', penalty='l2', max_iter=2000, 
        class_weight='balanced', random_state=args.random_state, n_jobs=-1
    )
    ,
    'KNN': KNeighborsClassifier(
        n_neighbors=5, 
        weights='distance', 
        algorithm='auto',
        leaf_size=100,
        n_jobs=-1         
    ),
    'MLP': MultiLayerPerceptron(
            input_features  = 640,
            hidden_features = 1024,
            output_classes  = 2,
            dropout_rate    = 0.2,
        ).to(DEVICE)
}
if args.classifier == "ENSEMBLE":
    with open(args.ensemblepath, "r") as file:
        ensemble_components = [l.strip() for l in file if l.strip()]
    
    if args.ensemblemode == "voting":
        selected_classifier = VotingClassifier(estimators=[(comp, models[comp]) for comp in ensemble_components], voting='hard', n_jobs=1)
    else:
        final_mdl =  RandomForestClassifier(
            n_estimators=100,      
            n_jobs=-1,             
            random_state=42,       
            class_weight='balanced' 
        )
    
        selected_classifier = StackingClassifier(estimators=[(comp, models[comp]) for comp in ensemble_components], final_estimator=final_mdl,
    cv=5, passthrough=True)  
else: 
    selected_classifier = models[args.classifier]



if args.classifier == "MLP":
    input_dim = X_train.shape[1]
    selected_classifier = MultiLayerPerceptron(
        input_features=input_dim,
        hidden_features=1024,
        output_classes=2,
        dropout_rate=0.2
    ).to(DEVICE)

    def make_loader(X, y, shuffle=False):
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=128, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val, y_val)
    test_loader  = make_loader(X_test, y_test)

    best_acc, _ = train(
        model=selected_classifier,
        num_epochs=50,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir="./checkpoints_mlp",
        model_name="MLP", 
        device=DEVICE
    )


    loss, acc, prec, rec, f1, auc = test(
        model=selected_classifier,
        test_loader=test_loader,
        model_name="MLP",
        device=DEVICE
    )

    metrics = {
        "classifier": args.classifier,
        "extractor": args.model,
        "auc": [auc],   
        "accuracy": [acc], 
        "precision": [prec], 
        "recall": [rec], 
        "f1-score": [f1]
    }
else:
    selected_classifier.fit(X_train, y_train)
    y_pred = selected_classifier.predict(X_test)
    y_proba = selected_classifier.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    metrics = {
        "classifier": args.classifier,
        "extractor": args.model,
        "auc": auc, 
        "accuracy": [report['accuracy']], 
            "precision": [report['macro avg']['precision']], 
            "recall": [report['macro avg']['recall']], 
            "f1-score": [report['macro avg']['f1-score']]}

df = pd.DataFrame(metrics)

results_path = "bert-extractor-metrics.csv"

exists = os.path.exists(results_path)
df.to_csv(results_path, index=False, mode='a', header= not exists)
