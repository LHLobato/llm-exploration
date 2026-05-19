from multimodal import Domain_Ensemble
from data import load_dataset_from_disk
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--image_path", type=str, help="images dataset")
parser.add_argument("--llm_path", type=str, help="pre-trained llm path")
parser.add_argument("--cnn_path", type=str, help="pre-trained cnn path")
parser.add_argument("--tab_path", type=str, help="pre-trained tabular model path")

args = parser.parse_args()


#IMAGE_PATH = "/home/lhslobato/novo-reshaping/images/GADF/val"
domain_df = pd.read_csv("../data/acme/val.csv")

IMG_PATH = args.img_path
LLM_PATH = args.llm_path
MLP_PATH = args.tab_path

print(len(load_dataset_from_disk(IMAGE_PATH)))
