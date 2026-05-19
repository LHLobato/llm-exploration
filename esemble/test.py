from multimodal import DomainEnsemble
from data import load_dataset_from_disk



IMAGE_PATH = "/home/lhslobato/novo-reshaping/images/GADF/val"
LLM_PATH = ""
MLP_PATH = ""

print(len(load_dataset_from_disk(IMAGE_PATH)))
