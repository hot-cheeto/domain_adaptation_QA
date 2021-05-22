import os
import random
from collections import defaultdict

import string
import itertools
import re
import time

import nltk
import numpy as np


def dict2string(dictionary, title):
    message = '\n\t'+title+'\n'
    info = ['\t\t' + str(key)+'=' + str(value) + '\n' for key,
            value in list(dictionary.items())]
    info_str = ''.join(info)
    message += info_str

    return message


def get_timestamp(format_='%m/%d/%y %I:%M:%S %p'):
    return time.strftime(format_, time.localtime(int(time.time())))



def dict2dataframe(rowid2data, example_id = 'example_id', notebook = False):
    
    df_dict = {}
    tqdm_pbar = tqdm_notebook if notebook else tqdm

    for k, v in tqdm_pbar(rowid2data.items()):
        
        if len(df_dict.keys()) < 1:
            headers = list(v.keys())
            df_dict = {kk: [] for kk in [example_id] + headers}

        df_dict[example_id].append(k)
        for kk, vv in v.items():
            df_dict[kk].append(vv)
            
    df = pd.DataFrame(df_dict)
    
    return df

