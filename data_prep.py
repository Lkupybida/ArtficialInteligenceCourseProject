import pandas as pd

def xlsb_to_csv_wrapper():
    df = pd.read_excel('data/original/Транзакції_електрозарядки_01.2021-10.2024.xlsb', sheet_name='Base_01_2021-10_2024')
    df = start_end_time(df, 'Період зарядки')
    df = true_falsify(df, ['Зарядка через адаптер', 'Fishka', 'Успішна'])
    df = translate_cols(df)
    df = df.drop(columns='ChargingTime')
    df = add_lat_lon(df, 'InternalNum')
    df = add_station_type(df, 'Station')
    df = charging_time(df, 'Start', 'End')
    df['Revenue'] = df['Revenue'].replace({'\xa0': '', ',': '.'}, regex=True)
    df['Revenue'] = df['Revenue'].astype(float)
    df.to_csv('data/original/dataset.csv', index=False)

def start_end_time(df, col):
    df[['Start', 'Finish']] = df[col].str.split('-', expand=True)
    df['Start'] = pd.to_datetime(df['Start'], format='%d.%m.%Y %H:%M:%S')
    df['Finish'] = pd.to_datetime(df['Finish'], format='%d.%m.%Y %H:%M:%S')
    df = df.drop(columns=[col])
    return df

def true_falsify(df, cols):
    for col in cols:
        df[col] = df[col].map({"Так": True, "Ні": False})
    return df

def translate_cols(df):
    column_mapping = {
        'Ідентифікатор сесії': 'SessionID',
        'Час зарядки': 'ChargingTime',
        'Місто': 'City',
        'Станція': 'Station',
        'Внутрішній номер': 'InternalNum',
        'Номер порту': 'PortNum',
        'Тип порту': 'PortType',
        'Зарядка через адаптер': 'Adapter',
        'Клієнт': 'Client',
        'Тариф, грн з ПДВ/кВт': 'Tariff',
        'кВт': 'kWh',
        'Максимальна потужність, кВт': 'MaxPower',
        'Виручка, грн з ПДВ': 'Revenue',
        'Fishka': 'Fishka',
        'Успішна': 'Successful',
        'Причина відключення': 'DisconnectionReason',
        'Start': 'Start',
        'Finish': 'End'
    }
    df = df.rename(columns=column_mapping)
    new_order = ['SessionID', 'Start', 'End', 'kWh', 'Revenue', 'InternalNum',
                 'ChargingTime', 'City', 'Station', 'PortNum', 'PortType', 'Adapter',
                 'Client', 'Tariff', 'MaxPower', 'Fishka', 'Successful', 'DisconnectionReason']
    df = df[new_order]
    return df


def add_lat_lon(df, col):
    locations = pd.read_csv('data/locations.csv')
    locations = locations.rename(columns={'Унікод АЗС': col})
    locations = locations[['Latitude', 'Longitude', col]]
    result = pd.merge(df, locations, on=col, how='left')
    result = result.drop(columns = [col])
    return result

def add_station_type(df, col):
    stations = pd.read_excel("data/original/Транзакції_електрозарядки_01.2021-10.2024.xlsb", engine="pyxlsb", sheet_name=1, usecols=['Станція', 'Тип станції'])
    stations = stations.rename(columns={'Станція': col, 'Тип станції': 'Type'})
    result = pd.merge(df, stations, on=col, how='left')
    # result = result.drop(columns = [col])
    return result

def create_type_dummies(df, col):
    df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    dummy_cols = [c for c in df.columns if c.startswith(f"{col}_")]
    df[dummy_cols] = df[dummy_cols].astype(int)
    
    return df

def charging_time(df, col1, col2):
    df[col1] = pd.to_datetime(df[col1])
    df[col2] = pd.to_datetime(df[col2])

    df['ChargingTime'] = round((df[col2] - df[col1]).dt.total_seconds() / 60, 2)
    return df

def clean_data_wrapper():
    df = pd.read_csv('data/original/dataset.csv')
    df = df.dropna(subset=['Successful','Adapter'])
    df = drop_unnecesary_data(df)
    df['Client'] = df['Client'].astype(str)
    df['PortType'] = df['PortType'].astype(str)
    df['Start'] = pd.to_datetime(df['Start'], format='%Y-%m-%d %H:%M:%S')
    df['End'] = pd.to_datetime(df['End'], format='%Y-%m-%d %H:%M:%S')
    df.to_csv('data/clean/dataset.csv', index_label='idx')

def drop_unnecesary_data(df):
    cols = ['Successful', 'City', 'DisconnectionReason', 'SessionID']
    df = df.drop(columns=cols)
    return df