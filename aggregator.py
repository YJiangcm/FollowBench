import os
import pandas as pd
import glob
import numpy as np


def aggregate(result_files):
    dfs = [pd.read_csv(file) for file in result_files]
    # print(type(dfs[0]['level 1'].iloc[0]))
    for df in dfs:
        df['score'] = (df['level 1'] + 2*df['level 2'] + 3*df['level 3'] + 4*df['level 4'] + 5*df['level 5']) / 15

    scores = np.mean([np.array(df['score']) for df in dfs], axis=0)
    return scores


evaluation_output_dir = "evaluation_result"
hsr_results = glob.glob(f"{evaluation_output_dir}/*hsr.csv")
ssr_results = glob.glob(f"{evaluation_output_dir}/*ssr.csv")

print(aggregate(hsr_results))
print(aggregate(ssr_results))