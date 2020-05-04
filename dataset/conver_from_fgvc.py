#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 6:48 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import json, os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--file",
        help="json file to be converted",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--root",
        help="root path to save image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sp",
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )

    args = parser.parse_args()
    return args

def convert(json_file, image_root):
    new_annos = []
    df = pd.read_csv(os.path.join(json_file))
    print("Converting file {} ...".format(json_file))
    for i in tqdm(range(len(df))):
        new_annos.append({"image_id": df.iloc[i].values[0],
                          "im_height": 1356,
                          "im_width": 2048,
                          "category_id": int(np.argmax(df.iloc[i].values[1:])),
                          "fpath":  os.path.join(image_root, 'images', df.iloc[i].values[0] + '.jpg')
                          })
    num_classes = 4
    return {"annotations": new_annos,
            "num_classes": num_classes}

if __name__ == "__main__":
    args = parse_args()
    converted_annos = convert(args.file, args.root)
    save_path = os.path.join(args.sp, "converted_" + os.path.split(args.file)[-1][:-3]+'json')
    print("Converted, Saveing converted file to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(converted_annos, f)