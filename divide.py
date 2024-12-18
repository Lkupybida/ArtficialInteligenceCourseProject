import pandas as pd
def train_val_test(df, month, col):
    df[col] = pd.to_datetime(df[col])
    if month == True:
        train = df[df[col] < '2024-09-01']
        valid = df[(df[col] < '2024-10-01') & (df[col] >= '2024-09-01')]
        test =  df[df[col] >= '2024-10-01']
    else:
        train = df[df[col] < '2024-10-25']
        valid = df[(df[col] < '2024-10-28') & (df[col] >= '2024-10-25')]
        test =  df[df[col] >= '2024-10-28']
        
    return train, valid, test
