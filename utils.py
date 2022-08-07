import numpy as np
import pandas as pd
from dask import dataframe as dd
from tqdm.autonotebook import tqdm
from typing import Union
from dask.distributed import Client

numerical_features = ['feature_'+str(i) for i in [2,3,4,10,11,12,15,19,23]]
features_to_impute = ['feature_'+str(i) for i in[5,6,7,13,14,17,18,20]]
features_to_drop = ['feature_'+str(i) for i in [1,8,21]]
label_col = 'label'

def load_data(use_dask:bool = False) -> Union[pd.DataFrame,dd.DataFrame]:
    deligate_read_parquet = pd.read_parquet
    deligate_to_numeric = pd.to_numeric
    if use_dask:
        deligate_read_parquet = dd.read_parquet
        deligate_to_numeric = dd.to_numeric
        client = Client(n_workers=4, threads_per_worker=2, memory_limit="16GB")
    df_1 = deligate_read_parquet('table_1.parquet').set_index('id')
    df_2 = deligate_read_parquet('table_2.parquet').set_index('id')
    df = df_1.join(df_2)
    del df_1
    del df_2
    df['label'] = deligate_to_numeric(df['label'])
    for feature in tqdm(numerical_features):
        df[feature] = deligate_to_numeric(df[feature])

    return df

def impute_low_frequencies(series:pd.Series,cutoff_threshold:float=0.1,cutoff_label:str = 'OTHER') -> pd.Series:
    value_counts = series.value_counts(normalize=True)
    values_to_keep = set(np.where(value_counts>=cutoff_threshold,value_counts.index,cutoff_label))
    return_series = series.apply(lambda x: x if x in values_to_keep else cutoff_label)
    return return_series

