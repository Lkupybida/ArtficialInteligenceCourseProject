import pandas as pd

stations = pd.read_csv('data\clean\hourly_uniformly_azk.csv')
def train_val_test(df, month):
    df['time'] = pd.to_datetime(df['time'])
    if month == True:
        train = df[df['time'] < '2024-09-01']
        valid = df[(df['time'] < '2024-10-01') & (df['time'] >= '2024-09-01')]
        test =  df[df['time'] >= '2024-10-01']
    else:
        train = df[df['time'] < '2024-10-25']
        valid = df[(df['time'] < '2024-10-28') & (df['time'] >= '2024-10-25')]
        test =  df[df['time'] >= '2024-10-28']
        
    return train, valid, test

train, val, test = train_val_test(stations, False)
print(train)