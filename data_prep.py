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
    df = df[pd.isna(df['Latitude']) == False]
    df.to_csv('data/clean/dataset.csv', index_label='idx')

def drop_unnecesary_data(df):
    cols = ['Successful', 'City', 'DisconnectionReason', 'SessionID']
    df = df.drop(columns=cols)
    return df

def create_hourly_dataset():
    df = pd.read_csv('data/clean/dataset.csv')
    new_df = pd.DataFrame()
    new_df['time'] = df['Start']
    new_df['kWh'] = df['kWh']
    new_df['time'] = pd.to_datetime(new_df['time'], format='%Y-%m-%d %H:%M:%S')
    new_df.set_index('time', inplace=True)
    hourly_df = new_df.resample('H').sum()
    hourly_df.reset_index(inplace=True)
    hourly_df.to_csv('data/clean/hourly.csv', index=False)


import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm


def convert_to_hourly_charging(df):
    min_time = df['Start'].min().floor('h')
    max_time = df['End'].max().ceil('h')
    hourly_data = []
    # with tqdm(total=len(df), desc="Processing Charging Data") as pbar:
    for _, row in df.iterrows():
        start_time = row['Start']
        end_time = row['End']
        total_kwh = row['kWh']
        duration = end_time - start_time
        if duration.total_seconds() < 1e-6:
            hourly_data.append({
                'time': start_time.floor('h'),
                'kWh': total_kwh
            })
            continue
        hours = pd.date_range(
            start=start_time.floor('h'),
            end=end_time,
            freq='h'
        )
        for hour in hours:
            hour_start = hour
            hour_end = hour + timedelta(hours=1)
            overlap_start = max(start_time, hour_start)
            overlap_end = min(end_time, hour_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600
            hour_kwh = (overlap_duration / (duration.total_seconds() / 3600)) * total_kwh
            hourly_data.append({
                'time': hour_start,
                'kWh': hour_kwh
            })
    hourly_df = pd.DataFrame(hourly_data)
    hourly_df = hourly_df.groupby('time')['kWh'].sum().reset_index()
    full_time_series = pd.DataFrame({
        'time': pd.date_range(start=min_time, end=max_time, freq='h')
    })
    result_df = full_time_series.merge(hourly_df, on='time', how='left').fillna(0)
    # result_df.to_csv(output_file, index=False)
    # print(f"Conversion complete. Output saved to {output_file}")
    return result_df

def convert_to_hourly_by_station(input_file):
    df = pd.read_csv(input_file, parse_dates=['Start', 'End'], low_memory=False)
    df_hourly_azk = pd.DataFrame()
    azk_dfs_list = []
    hapes= 0
    with tqdm(total=len(df['Latitude'].unique()), desc="Processing AZK") as pbar:
        for azk_num in df['Latitude'].unique():
            df_azk = pd.DataFrame()
            df_azk_hourly = pd.DataFrame()
            df_azk = df[df['Latitude'] == azk_num]
            df_azk_hourly = convert_to_hourly_charging(df_azk)
            df_azk_hourly['Latitude'] = azk_num
            df_azk_hourly['Longitude'] = df_azk['Longitude'].values[0]
            azk_dfs_list.append(df_azk_hourly)
            hapes = hapes + df_azk_hourly.shape[0]
            pbar.update(1)
    df_hourly_azk = pd.concat(azk_dfs_list, axis=0)

    df_hourly_azk.to_csv('data/clean/hourly_uniformly_azk.csv', index=False)
