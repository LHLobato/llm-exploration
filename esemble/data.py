import pandas as pd
from pathlib import Path


def load_dataset_from_disk(image_path: str) -> list:

    base = Path(image_path)

    malicious_files = [f.stem for f in (base / 'malicious').iterdir() if f.is_file()]
    benign_files    = [f.stem for f in (base / 'benign').iterdir() if f.is_file()]

    all_files = malicious_files + benign_files
    all_domains = [s.split('-', 1)[-1] for s in all_files]

    domain_df = pd.read_csv("../data/acme/val.csv")
    valid_domain = domain_df[domain_df['name'].isin(all_domains)]
    return valid_domain
