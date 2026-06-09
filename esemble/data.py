import pandas as pd
from pathlib import Path

def load_dataset_from_disk(image_path: str, domain_df: pd.DataFrame):
    base = Path(image_path)
    EXTENSIONS = {'.png', '.jpg', '.jpeg'}
    label_map = {'malicious': 1, 'normal': 0}

    rows = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        label = label_map.get(folder.name.lower(), -1)
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in EXTENSIONS:
                domain = f.stem.split('-', 1)[-1]
                rows.append({'image_path': f, 'name': domain, 'label': label})

    files_df = pd.DataFrame(rows)

    merged = files_df.merge(domain_df, on='name', how='inner').drop_duplicates()

    print(f"[IMAGES]       {len(files_df)}")
    print(f"[DOMAINS]      {len(domain_df)}")
    print(f"[MATCHED]      {len(merged)}")
    print(f"[SUCCESS-RATE] {len(merged)/len(domain_df):.2%}")

    return merged
