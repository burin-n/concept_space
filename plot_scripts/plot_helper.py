import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
import os
import numpy as np

model2markers = {
    # "Full": "P",
    "Ambient": "P",
    "RAND": "o", 
    "MLR": "*", 
    "LDA": "p", 
    "CPCA": ">", 
    "COV": "<", 
    "LEACE": "^",
    # "WORST": "x", 
    "BEST": "+",
}

model2color = {
    k: f"C{i}" for i,k in enumerate(model2markers.keys())
}
model2color["BEST"] = "black"



def sort_key(x):    
    sort_key_list = ["WORST", "BEST", "EYE", "Full", "RANDOM", "RAND", "CLF", "CLFB", "MLR", "MLRb", "LC", "LCb", "SR", "LDA", "PCA", "CPCA", "COV", "WCOV", "LEACE", "WLEACE"]
    sort_key_ = {k:i for i,k in enumerate(sort_key_list)}
    return_key = []
    for elem in x:        
        if elem in sort_key_:
            return_key.append(sort_key_[elem])
        else:
            return_key.append(elem)
    return return_key

def get_best_val(EYE_df, concepts, mode=None):
    if mode is not None:
        EYE_df = EYE_df[EYE_df["mode"] == mode]
    sel_df_c0 = EYE_df[EYE_df["concept"] == concepts[0]]
    sel_df_c1 = EYE_df[EYE_df["concept"] == concepts[1]]    
    reten0 = sel_df_c0["retention"].iloc[0].item()
    leak0 = sel_df_c0["leakage"].iloc[0].item()
    pur0 = 1 - sel_df_c1["leakage"].iloc[0].item()
    inter0 = 1 - sel_df_c1["retention"].iloc[0].item()
    
    reten1 = sel_df_c1["retention"].iloc[0].item()
    leak1 = sel_df_c1["leakage"].iloc[0].item()
    pur1 = 1 - sel_df_c0["leakage"].iloc[0].item()
    inter1 = 1 - sel_df_c0["retention"].iloc[0].item()
    
    return {
        "concept": [concepts[0], concepts[1]],
        "concept1": [concepts[1], concepts[0]],
        "retention": [reten0, reten1],
        "leakage": [leak0, leak1],
        "purity": [pur0, pur1],
        "interference": [inter0, inter1]
        # "purity": [1-pur0, 1-pur1],
        # "interference": [1-inter0, 1-inter1]
    }
    
def get_worst_val(EYE_df, concepts, mode=None):
    if mode is not None:
        EYE_df = EYE_df[EYE_df["mode"] == mode]
        
    sel_df_c0 = EYE_df[EYE_df["concept"] == concepts[0]]
    sel_df_c1 = EYE_df[EYE_df["concept"] == concepts[1]]

    reten0 = sel_df_c0["leakage"].iloc[0].item()
    leak0 = sel_df_c0["retention"].iloc[0].item()
    pur0 = 1 - sel_df_c1["retention"].iloc[0].item()
    inter0 = 1 - sel_df_c1["leakage"].iloc[0].item()
    
    reten1 = sel_df_c1["leakage"].iloc[0].item()
    leak1 = sel_df_c1["retention"].iloc[0].item()
    pur1 = 1 - sel_df_c0["retention"].iloc[0].item()
    inter1 = 1 - sel_df_c0["leakage"].iloc[0].item()
    
    return {
        "concept": [concepts[0], concepts[1]],
        "concept1": [concepts[1], concepts[0]],
        "retention": [reten0, reten1],
        "leakage": [leak0, leak1],
        "purity": [pur0, pur1],
        "interference": [inter0, inter1]
        # "purity": [1-pur0, 1-pur1],
        # "interference": [1-inter0, 1-inter1]
    }
    
def merge_best_val(df_row, key, norm=False, mode="freezed"):
    c0 = df_row["concept"]
    c1 = df_row["concept1"]
    if c0 == best_val["concept"][0] and c1 == best_val["concept1"][0]:
        idx = 0
    elif c0 == best_val["concept"][1] and c1 == best_val["concept1"][1]:
        idx = 1
    else:
        raise ValueError()
    if key in ["purity", "interference"]:
        return f"{100-100*df_row[key]:.1f}/{100-100*best_val[key][idx]:.1f}"
    else:
        return f"{100*df_row[key]:.1f}/{100*best_val[key][idx]:.1f}"


def inject_best_worst(df_in, MODE=None, drop_EYE_rows=False):
    df = df_in.copy()
    concept_uniq = df["concept"].unique()
    df["concept1"] = df["concept"].map(lambda x: concept_uniq[0] if x!=concept_uniq[0] else concept_uniq[1])
    best_val = get_best_val(df[df["method"] == "EYE"], concept_uniq, mode=MODE)
    worst_val = get_worst_val(df[df["method"] == "EYE"], concept_uniq, mode=MODE)
    
    df["purity"] = 1 - df["purity"]
    df["interference"] = 1 - df["interference"]
    
    best_df = pd.DataFrame({
        "concept": [concept_uniq[0], concept_uniq[1]],
        "method": ["BEST", "BEST"],
        "mode": [MODE, MODE],
        "seed": [0, 0],
        "retention": [best_val["retention"][0], best_val["retention"][1]],
        "purity": [best_val["purity"][0], best_val["purity"][1]],
        "leakage": [best_val["leakage"][0], best_val["leakage"][1]],
        "interference": [best_val["interference"][0], best_val["interference"][1]],
    })
    worst_df = pd.DataFrame({
        "concept": [concept_uniq[0], concept_uniq[1]],
        "method": ["WORST", "WORST"],
        "mode": [MODE, MODE],
        "seed": [0, 0],
        "retention": [worst_val["retention"][0], worst_val["retention"][1]],
        "purity": [worst_val["purity"][0], worst_val["purity"][1]],
        "leakage": [worst_val["leakage"][0], worst_val["leakage"][1]],
        "interference": [worst_val["interference"][0], worst_val["interference"][1]],
    })
    df.loc[df["method"] == "RANDOM", "method"] = "RAND"
    df.loc[df["method"] == "CLF", "method"] = "MLR"
    df.loc[df["method"] == "LC", "method"] = "MLR"
    df.loc[df["method"] == "PCA", "method"] = "CPCA"

    if drop_EYE_rows:
        df = df[ df["method"] != "EYE" ]
    
    df = pd.concat([worst_df, best_df , df], axis=0)
    df = df.map(lambda x: round(x*100,1) if type(x) == float else x)
    df = df.drop(columns=["seed", "concept1"])
    df = df.sort_values(by=["method", "concept"], key=sort_key)
    return df

def get_DF(in_path, MODE):
    df = pd.read_csv(in_path)
    df = df.sort_values(by=["method", "concept"], key=sort_key)
    df.columns = ['concept', 'method', 'mode', 'seed', 'retention',
           'purity', 'leakage', 'interference', 'seed_conf',
           'retention_conf', 'purity_conf', 'leakage_conf', 'interference_conf']
    df = df[df["mode"] == MODE]
    return df