import pandas as pd
import os
import numpy as np
from pdb import set_trace as bp
import sys
from pathlib import Path

"""Calculate means of result files."""

def sort_dfs(dfs):
    res = []
    for df in dfs:
        start = df.iloc[:4,:].sort_values(by=[' method'], ascending=False)
        end = df.iloc[4:,:].sort_values(by=[' method'], ascending=False)
        df_new = pd.concat([start, end])
        df_new.set_index(np.arange(len(df)), inplace=True)
        res.append(df_new)
    return res

if __name__ == '__main__':
    name = sys.argv[1]
    output_dir = sys.argv[2]
    if name not in ['small', 'go', 'xml']:
        print('invalid dataset name')
        sys.exit(1)

    dfs = [pd.read_csv(Path(output_dir, fname), index_col=False) for fname in os.listdir(output_dir) if 'csv' in fname and name in fname and 'mean' not in fname]
    print(len(dfs))

    if name == 'xml':
        dfs = sort_dfs(dfs)

    df = pd.concat(dfs).groupby(level=0).mean()

    df.insert(0, 'method', dfs[0].values[:,1])
    df.insert(0, 'dataset', dfs[0].values[:,0])
    df.to_csv(Path(output_dir, f'mean_scores_{name}.csv'), index=False, float_format='%.4f')
    # df.to_csv(Path(path, f'mean_scores_{name}.csv'), index=False)






