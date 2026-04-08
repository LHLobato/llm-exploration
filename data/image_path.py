import pandas as pd

train_prompt = pd.read_csv("acme/train.csv", index_col=False)
val_prompt   = pd.read_csv("acme/val.csv",   index_col=False)
test_prompt  = pd.read_csv("acme/test.csv",  index_col=False)

train_img = pd.read_csv("/home/lhslobato/novo-reshaping/image_generation/filenames-GADF-train.csv", index_col=False)
val_img   = pd.read_csv("/home/lhslobato/novo-reshaping/image_generation/filenames-GADF-val.csv",   index_col=False)
test_img  = pd.read_csv("/home/lhslobato/novo-reshaping/image_generation/filenames-GADF-test.csv",  index_col=False)

def merge_and_check(prompt_df, img_df, name):
    merged = pd.merge(prompt_df, img_df[['name', 'image_path']],
                      left_on='name', right_on='name', how='left')
    missing = merged['image_path'].isna().sum()
    print(f"{name}: {len(merged)} rows, {missing} missing image_path ({100*missing/len(merged):.1f}%)")
    return merged

merged_train = merge_and_check(train_prompt, train_img, "train")
merged_val   = merge_and_check(val_prompt,   val_img,   "val")
merged_test  = merge_and_check(test_prompt,  test_img,  "test")

merged_train.to_csv("vlm-train.csv", index=False)
merged_val.to_csv("vlm-val.csv",     index=False)
merged_test.to_csv("vlm-test.csv",   index=False)