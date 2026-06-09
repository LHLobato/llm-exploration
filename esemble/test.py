import pandas as pd
from multimodal import Domain_Ensemble, tprint
from data import load_dataset_from_disk
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score

parser = ArgumentParser()
parser.add_argument("--train_img_path", type=str)
parser.add_argument("--test_img_path", type=str)
parser.add_argument("--llm_path", type=str)
parser.add_argument("--cnn_path", type=str)
parser.add_argument("--tab_path", type=str)
args = parser.parse_args()

FEATURE_COLS_TO_DROP = ["name", "malicious", "prompt", "image_path", "label"]

def main():
    domain_df = pd.read_csv("../data/acme/val.csv")
    domain_test_df = pd.read_csv("../data/acme/test.csv")

    train_df = load_dataset_from_disk(args.train_img_path, domain_df)
    test_df  = load_dataset_from_disk(args.test_img_path, domain_test_df)

    scaler = MinMaxScaler()
    tab_cols = [c for c in train_df.columns if c not in FEATURE_COLS_TO_DROP]

    scaler = MinMaxScaler()
    train_df[tab_cols] = scaler.fit_transform(train_df[tab_cols])
    test_df[tab_cols]  = scaler.transform(test_df[tab_cols])



    train_prompts = train_df['prompt'].values
    test_prompts  = test_df['prompt'].values

    train_labels = train_df['label'].values
    test_labels  = test_df['label'].values

    ensemble = Domain_Ensemble(
        cnn_path=args.cnn_path,
        tab_path=args.tab_path,
        llm_path=args.llm_path
    )


    ensemble.fit(
        train_df['image_path'].tolist(),
        train_df,          
        train_prompts,
        train_labels
    )
    preds = ensemble.predict(
        test_df['image_path'].tolist(),
        test_df,
        test_prompts
    )

    
    tprint("[REPORT] :")
    tprint(classification_report(test_labels, preds))

    proba = ensemble.predict_proba(
        test_df['image_path'].tolist(),
        test_df,
        test_prompts
    )

    tprint("[AUC]: ")
    tprint(roc_auc_score(test_labels, proba[:, 1]))
    tprint("[META-MODEL COEFFICIENTS]:")
    feature_names = ["CNN", "MLP", "LLM"]
    coefs = ensemble.meta_model.coef_[0]
    for name, coef in zip(feature_names, coefs):
        tprint(f"  {name}: {coef:.4f}")
    tprint(f"  Intercept: {ensemble.meta_model.intercept_[0]:.4f}")

if __name__ == "__main__":
    main()
