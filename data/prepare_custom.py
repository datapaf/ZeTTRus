from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from transformers import HfArgumentParser
import os
import math
from typing import List, Optional

OUT_TRAIN_DIR = './train'
OUT_VALID_DIR = './valid'
LANG = 'ru_en'


def process_text(text):
    return text.replace("\\t", "\t").replace("\\n", "\n").replace("\\_", "\\")


def process_data_split(dset, split_name):
    result = []
    print(f"Processing {split_name} data...")
    for item in tqdm(dset[split_name]):
        text = item['text'][0]['content'] + " " + item['text'][1]['content']
        text = process_text(text)
        result.append(text)
    return result

if __name__ == "__main__":

    out_train_dir = Path(OUT_TRAIN_DIR)
    out_valid_dir = Path(OUT_VALID_DIR)

    out_train_dir.mkdir(exist_ok=True, parents=True)
    out_valid_dir.mkdir(exist_ok=True, parents=True)

    dset = load_dataset("ikkiren/big_dataset")
    
    train_data = process_data_split(dset, 'train')
    valid_data = process_data_split(dset, 'test')
    
    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text"]
    train_df.to_parquet(out_train_dir / f"{LANG}.parquet")

    valid_df = pd.DataFrame(valid_data)
    valid_df.columns = ["text"]
    valid_df.to_parquet(out_valid_dir / f"{LANG}.parquet")
