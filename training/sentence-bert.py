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
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, 
                   help="Informe a arquitetura do modelo")
parser.add_argument("--classifier", type=str, default='RF', choices = ["RF", "XGB", "SVM", "LR", "MLP", "ENSEMBLE", "KNN"], help="Machine Learning Model Classifier")
parser.add_argument("--fitsentencemodel", action="store_true", help="Fine tuning")
parser.add_argument("--ensemblepath", type=str, help="Caminho para o arquivo de estrutura do ensemble")
parser.add_argument("--ensemblemode", type=str, default="stacking", help="Modo de ensemble")
parser.add_argument("--fractais", action="store_true", help="Features Fractais")
parser.add_argument("--featondisk", action="store_true", help="Flag")
parser.add_argument("--random_state", type=int, default=42)

args = parser.parse_args()
model_name = args.model
model = SentenceTransformer(model_name, device='cuda')
prefix = "Represent this domain name for malware detection: "

domain_names = pd.read_csv("../dns-feature-enrichment/csvs/dataset.csv", index_col=False)['name'].values
labels = pd.read_csv("../dns-feature-enrichment/csvs/dataset.csv", index_col=False)['malicious'].values

def extract_snowflake_features(domain_list, model, batch_size=128):
    processed_texts = [prefix + d for d in domain_list]
    
    embeddings = model.encode(
        processed_texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    return embeddings

state = 0 



X_text_train, X_text_temp, y_train, y_temp = train_test_split(
    domain_names, labels, test_size=0.3, random_state=state, stratify=labels
)
X_text_val, X_text_test, y_val, y_test = train_test_split(
    X_text_temp, y_temp, test_size=0.5, random_state=state, stratify=y_temp
)


if args.fitsentencemodel:

    def create_pairs(texts, labels):
        pairs = []
        mal_idx = np.where(labels == 1)[0]
        ben_idx = np.where(labels == 0)[0]

        for i in range(0, len(mal_idx)-1, 2):
            pairs.append(InputExample(texts=[texts[mal_idx[i]], texts[mal_idx[i+1]]]))
        
        # Criando pares benignos
        for i in range(0, len(ben_idx)-1, 2):
            pairs.append(InputExample(texts=[texts[ben_idx[i]], texts[ben_idx[i+1]]]))
        
        return pairs

    X_text_snowflake, X_text_clf_train, y_snowflake, y_clf_train = train_test_split(
        X_text_train, y_train, test_size=0.5, random_state=state, stratify=y_train
        )


    train_examples = create_pairs(X_text_snowflake, y_snowflake)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=3,
        warmup_steps=100,
        optimizer_params={'lr': 5e-5},
        output_path=f"./{args.model}"
    )


    X_train = extract_snowflake_features(X_text_clf_train, model)
    X_val = extract_snowflake_features(X_text_val, model)
    X_test = extract_snowflake_features(X_text_test, model)

    y_train = y_clf_train
else:
    
    X_train = extract_snowflake_features(X_text_train, model)
    X_val = extract_snowflake_features(X_text_val, model)
    X_test = extract_snowflake_features(X_text_test, model)


args = parser.parse_args()
#if not args.modelpath:
#    raise Exception("There must be a modelpath to load")

#if not args.image_folder:
  #  raise Exception("There must be a imagefolder to load")

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
        loss='hinge', penalty='l2', max_iter=2000, 
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
        hidden_features=256,
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
        "auc": [auc], 
        "accuracy": [report['accuracy']], 
            "precision": [report['macro avg']['precision']], 
            "recall": [report['macro avg']['recall']], 
            "f1-score": [report['macro avg']['f1-score']]}

df = pd.DataFrame(metrics)

results_path = "bert-extractor-metrics.csv"

exists = os.path.exists(results_path)
df.to_csv(results_path, index=False, mode='a', header= not exists)
