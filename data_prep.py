import os
from sklearn.metrics import mean_absolute_percentage_error as calc_mape
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import pytz
import time


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

def convert_to_hourly_by_station():
    df = pd.read_csv('data/clean/dataset.csv', parse_dates=['Start', 'End'], low_memory=False)
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
            df_azk_hourly['InternalNum'] = df_azk['InternalNum'].values[0]
            azk_dfs_list.append(df_azk_hourly)
            hapes = hapes + df_azk_hourly.shape[0]
            pbar.update(1)
    df_hourly_azk = pd.concat(azk_dfs_list, axis=0)

    df_hourly_azk.to_csv('data/clean/hourly_uniformly_azk.csv', index=False)

def add_weather_data():
    df = pd.read_csv('data/clean/dataset.csv', parse_dates=['Start', 'End'], low_memory=False)

    azk_dfs_list = []
    with tqdm(total=len(df['Latitude'].unique()), desc="Processing AZK") as pbar:
        for azk_num in df['Latitude'].unique():
            df_azk = pd.DataFrame()
            df_azk_hourly = pd.DataFrame()
            df_azk = df[df['Latitude'] == azk_num]
            df_azk_hourly = convert_to_hourly_charging(df_azk)
            df_azk_hourly['Latitude'] = azk_num
            df_azk_hourly['Longitude'] = df_azk['Longitude'].values[0]
            df_azk_hourly['InternalNum'] = df_azk['InternalNum'].values[0]
            azk_dfs_list.append(df_azk_hourly)
            pbar.update(1)
            # break
    weathered_dfs_list = []
    with tqdm(total=len(df['Latitude'].unique()), desc="Adding weather") as pbar:
        for difi in azk_dfs_list:
            try:
                start_date = difi['time'].values[0]
                end_date = difi['time'].values[difi.shape[0]-1]

                start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

                cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)

                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": difi['Latitude'].values[0],
                    "longitude": difi['Longitude'].values[0],
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": ["temperature_2m", "dew_point_2m",
                               "apparent_temperature", "relative_humidity_2m",
                               "precipitation", "showers",
                               "snow_depth", "weathercode",
                               "cloudcover", "windspeed_10m",
                               "winddirection_10m", "shortwave_radiation",
                               "direct_radiation", "surface_pressure",
                               "visibility",
                               "rain", "snowfall"],
                    "timezone": "auto"
                }

                responses = openmeteo.weather_api(url, params=params)

                response = responses[0]
                # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
                # print(f"Elevation {response.Elevation()} m asl")
                # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
                # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
                hourly_rain = hourly.Variables(2).ValuesAsNumpy()
                hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
                hourly_apparent_temperature = hourly.Variables(4).ValuesAsNumpy()
                hourly_relative_humidity_2m = hourly.Variables(5).ValuesAsNumpy()
                hourly_precipitation = hourly.Variables(6).ValuesAsNumpy()
                hourly_showers = hourly.Variables(7).ValuesAsNumpy()
                hourly_snow_depth = hourly.Variables(8).ValuesAsNumpy()
                hourly_weathercode = hourly.Variables(9).ValuesAsNumpy()
                hourly_cloudcover = hourly.Variables(10).ValuesAsNumpy()
                hourly_windspeed_10m = hourly.Variables(11).ValuesAsNumpy()
                hourly_winddirection_10m = hourly.Variables(12).ValuesAsNumpy()
                hourly_shortwave_radiation = hourly.Variables(13).ValuesAsNumpy()
                hourly_direct_radiation = hourly.Variables(14).ValuesAsNumpy()
                hourly_surface_pressure = hourly.Variables(15).ValuesAsNumpy()
                hourly_visibility = hourly.Variables(16).ValuesAsNumpy()

                utc_time = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )

                local_tz = pytz.timezone(response.Timezone())
                local_time = utc_time.tz_convert(local_tz).tz_localize(None)

                hourly_data = {
                    "date": local_time,
                    "temperature_2m": hourly_temperature_2m,
                    "dew_point_2m": hourly_dew_point_2m,
                    "rain": hourly_rain,
                    "snowfall": hourly_snowfall,
                    "apparent_temperature": hourly_apparent_temperature,
                    "relative_humidity_2m": hourly_relative_humidity_2m,
                    "precipitation": hourly_precipitation,
                    "showers": hourly_showers,
                    "snow_depth": hourly_snow_depth,
                    "weathercode": hourly_weathercode,
                    "cloudcover": hourly_cloudcover,
                    "windspeed_10m": hourly_windspeed_10m,
                    "winddirection_10m": hourly_winddirection_10m,
                    "shortwave_radiation": hourly_shortwave_radiation,
                    "surface_pressure": hourly_surface_pressure,
                    "visibility": hourly_visibility
                }

                hourly_dataframe = pd.DataFrame(data=hourly_data)
                difi['time'] = pd.to_datetime(difi['time'])
                hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
                azk_weathered = pd.merge(
                    difi.rename(columns={"time": "datetime"}),
                    hourly_dataframe.rename(columns={"date": "datetime"}),
                    on="datetime",
                    how="inner"
                )

                weathered_dfs_list.append(azk_weathered)
                azk_weathered.to_csv(f"data/clean/azk/{difi['InternalNum'].values[0]}.csv", index=False)
                pbar.update(1)
            except Exception as e:
                time.sleep(60)
                start_date = difi['time'].values[0]
                end_date = difi['time'].values[difi.shape[0] - 1]

                start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

                cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)

                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": difi['Latitude'].values[0],
                    "longitude": difi['Longitude'].values[0],
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": ["temperature_2m", "dew_point_2m",
                               "apparent_temperature", "relative_humidity_2m",
                               "precipitation", "showers",
                               "snow_depth", "weathercode",
                               "cloudcover", "windspeed_10m",
                               "winddirection_10m", "shortwave_radiation",
                               "direct_radiation", "surface_pressure",
                               "visibility",
                               "rain", "snowfall"],
                    "timezone": "auto"
                }

                responses = openmeteo.weather_api(url, params=params)

                response = responses[0]
                # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
                # print(f"Elevation {response.Elevation()} m asl")
                # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
                # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
                hourly_rain = hourly.Variables(2).ValuesAsNumpy()
                hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
                hourly_apparent_temperature = hourly.Variables(4).ValuesAsNumpy()
                hourly_relative_humidity_2m = hourly.Variables(5).ValuesAsNumpy()
                hourly_precipitation = hourly.Variables(6).ValuesAsNumpy()
                hourly_showers = hourly.Variables(7).ValuesAsNumpy()
                hourly_snow_depth = hourly.Variables(8).ValuesAsNumpy()
                hourly_weathercode = hourly.Variables(9).ValuesAsNumpy()
                hourly_cloudcover = hourly.Variables(10).ValuesAsNumpy()
                hourly_windspeed_10m = hourly.Variables(11).ValuesAsNumpy()
                hourly_winddirection_10m = hourly.Variables(12).ValuesAsNumpy()
                hourly_shortwave_radiation = hourly.Variables(13).ValuesAsNumpy()
                hourly_direct_radiation = hourly.Variables(14).ValuesAsNumpy()
                hourly_surface_pressure = hourly.Variables(15).ValuesAsNumpy()
                hourly_visibility = hourly.Variables(16).ValuesAsNumpy()

                utc_time = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )

                local_tz = pytz.timezone(response.Timezone())
                local_time = utc_time.tz_convert(local_tz).tz_localize(None)

                hourly_data = {
                    "date": local_time,
                    "temperature_2m": hourly_temperature_2m,
                    "dew_point_2m": hourly_dew_point_2m,
                    "rain": hourly_rain,
                    "snowfall": hourly_snowfall,
                    "apparent_temperature": hourly_apparent_temperature,
                    "relative_humidity_2m": hourly_relative_humidity_2m,
                    "precipitation": hourly_precipitation,
                    "showers": hourly_showers,
                    "snow_depth": hourly_snow_depth,
                    "weathercode": hourly_weathercode,
                    "cloudcover": hourly_cloudcover,
                    "windspeed_10m": hourly_windspeed_10m,
                    "winddirection_10m": hourly_winddirection_10m,
                    "shortwave_radiation": hourly_shortwave_radiation,
                    "surface_pressure": hourly_surface_pressure,
                    "visibility": hourly_visibility
                }

                hourly_dataframe = pd.DataFrame(data=hourly_data)
                difi['time'] = pd.to_datetime(difi['time'])
                hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
                azk_weathered = pd.merge(
                    difi.rename(columns={"time": "datetime"}),
                    hourly_dataframe.rename(columns={"date": "datetime"}),
                    on="datetime",
                    how="inner"
                )
                weathered_dfs_list.append(azk_weathered)
                azk_weathered.to_csv(f"data/clean/azk/{difi['InternalNum'].values[0]}.csv", index=False)
                pbar.update(1)
            # break
    df_weathered = pd.concat(weathered_dfs_list, axis=0)
    df_weathered.to_csv('data/clean/weathered.csv', index=False)

def concat_csv_files(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    dataframes = [pd.read_csv(os.path.join(directory, file), index_col=0, header=0) for file in csv_files]
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    return combined_df

def optimized_weathered_df():
    df = combine_csv_parts()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['kWh'] = df['kWh'].astype(np.float32)
    df['Latitude'] = df['Latitude'].astype(np.float32)
    df['Longitude'] = df['Longitude'].astype(np.float32)
    df['InternalNum'] = df['InternalNum'].astype(np.int32)
    df['InternalNum'] = pd.Categorical(df['InternalNum'])
    df['temperature_2m'] = df['temperature_2m'].astype(np.float32)
    df['dew_point_2m'] = df['dew_point_2m'].astype(np.float32)
    df['rain'] = df['rain'].astype(np.float32)
    df['snowfall'] = df['snowfall'].astype(np.float32)
    df['apparent_temperature'] = df['apparent_temperature'].astype(np.float16)
    df.drop(columns=['relative_humidity_2m', 'precipitation'])
    df['showers'] = df['showers'].astype(np.float16)
    df['snow_depth'] = df['snow_depth'].astype(np.float16)
    df['weathercode'] = df['weathercode'].astype(np.float32)
    df['weathercode'] = pd.Categorical(df['weathercode'])
    df['cloudcover'] = df['cloudcover'].astype(np.float16)
    df['winddirection_10m'] = df['winddirection_10m'].astype(np.float16)
    df['shortwave_radiation'] = df['shortwave_radiation'].astype(np.float32)
    df['surface_pressure'] = df['surface_pressure'].astype(np.float16)
    df['visibility'] = df['visibility'].astype(np.float16)
    return df

def add_weather_data_outages():
    df = pd.read_csv('data/clean/outages.csv', parse_dates=['ДатаЧас'], low_memory=False)
    azk_dfs_list = []
    with tqdm(total=len(df['Унікод АЗС'].unique()), desc="Processing AZK") as pbar:
        for azk_num in df['Унікод АЗС'].unique():
            df_azk = pd.DataFrame()
            df_azk = df[df['Унікод АЗС'] == azk_num]
            azk_dfs_list.append(df_azk)
            pbar.update(1)
            # break
    weathered_dfs_list = []
    with tqdm(total=len(df['Унікод АЗС'].unique()), desc="Adding weather") as pbar:
        for difi in azk_dfs_list:
            try:
                start_date = difi['ДатаЧас'].values[0]
                end_date = difi['ДатаЧас'].values[difi.shape[0]-1]

                start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

                cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)

                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": difi['Latitude'].values[0],
                    "longitude": difi['Longitude'].values[0],
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": ["temperature_2m", "dew_point_2m",
                               "apparent_temperature", "relative_humidity_2m",
                               "precipitation", "showers",
                               "snow_depth", "weathercode",
                               "cloudcover", "windspeed_10m",
                               "winddirection_10m", "shortwave_radiation",
                               "direct_radiation", "surface_pressure",
                               "visibility",
                               "rain", "snowfall"],
                    "timezone": "auto"
                }

                responses = openmeteo.weather_api(url, params=params)

                response = responses[0]
                # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
                # print(f"Elevation {response.Elevation()} m asl")
                # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
                # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
                hourly_rain = hourly.Variables(2).ValuesAsNumpy()
                hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
                hourly_apparent_temperature = hourly.Variables(4).ValuesAsNumpy()
                hourly_relative_humidity_2m = hourly.Variables(5).ValuesAsNumpy()
                hourly_precipitation = hourly.Variables(6).ValuesAsNumpy()
                hourly_showers = hourly.Variables(7).ValuesAsNumpy()
                hourly_snow_depth = hourly.Variables(8).ValuesAsNumpy()
                hourly_weathercode = hourly.Variables(9).ValuesAsNumpy()
                hourly_cloudcover = hourly.Variables(10).ValuesAsNumpy()
                hourly_windspeed_10m = hourly.Variables(11).ValuesAsNumpy()
                hourly_winddirection_10m = hourly.Variables(12).ValuesAsNumpy()
                hourly_shortwave_radiation = hourly.Variables(13).ValuesAsNumpy()
                hourly_direct_radiation = hourly.Variables(14).ValuesAsNumpy()
                hourly_surface_pressure = hourly.Variables(15).ValuesAsNumpy()
                hourly_visibility = hourly.Variables(16).ValuesAsNumpy()

                utc_time = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )

                local_tz = pytz.timezone(response.Timezone())
                local_time = utc_time.tz_convert(local_tz).tz_localize(None)

                hourly_data = {
                    "date": local_time,
                    "temperature_2m": hourly_temperature_2m,
                    "dew_point_2m": hourly_dew_point_2m,
                    "rain": hourly_rain,
                    "snowfall": hourly_snowfall,
                    "apparent_temperature": hourly_apparent_temperature,
                    "relative_humidity_2m": hourly_relative_humidity_2m,
                    "precipitation": hourly_precipitation,
                    "showers": hourly_showers,
                    "snow_depth": hourly_snow_depth,
                    "weathercode": hourly_weathercode,
                    "cloudcover": hourly_cloudcover,
                    "windspeed_10m": hourly_windspeed_10m,
                    "winddirection_10m": hourly_winddirection_10m,
                    "shortwave_radiation": hourly_shortwave_radiation,
                    "surface_pressure": hourly_surface_pressure,
                    "visibility": hourly_visibility
                }

                hourly_dataframe = pd.DataFrame(data=hourly_data)
                difi['ДатаЧас'] = pd.to_datetime(difi['ДатаЧас'])
                hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
                azk_weathered = pd.merge(
                    difi.rename(columns={"ДатаЧас": "datetime"}),
                    hourly_dataframe.rename(columns={"date": "datetime"}),
                    on="datetime",
                    how="inner"
                )

                weathered_dfs_list.append(azk_weathered)
                azk_weathered.to_csv(f"data/clean/azk/outages/{difi['Унікод АЗС'].values[0]}.csv", index=False)
                pbar.update(1)
            except Exception as e:
                time.sleep(60)
                start_date = difi['ДатаЧас'].values[0]
                end_date = difi['ДатаЧас'].values[difi.shape[0] - 1]

                start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

                cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)

                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": difi['Latitude'].values[0],
                    "longitude": difi['Longitude'].values[0],
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": ["temperature_2m", "dew_point_2m",
                               "apparent_temperature", "relative_humidity_2m",
                               "precipitation", "showers",
                               "snow_depth", "weathercode",
                               "cloudcover", "windspeed_10m",
                               "winddirection_10m", "shortwave_radiation",
                               "direct_radiation", "surface_pressure",
                               "visibility",
                               "rain", "snowfall"],
                    "timezone": "auto"
                }

                responses = openmeteo.weather_api(url, params=params)

                response = responses[0]
                # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
                # print(f"Elevation {response.Elevation()} m asl")
                # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
                # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
                hourly_rain = hourly.Variables(2).ValuesAsNumpy()
                hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
                hourly_apparent_temperature = hourly.Variables(4).ValuesAsNumpy()
                hourly_relative_humidity_2m = hourly.Variables(5).ValuesAsNumpy()
                hourly_precipitation = hourly.Variables(6).ValuesAsNumpy()
                hourly_showers = hourly.Variables(7).ValuesAsNumpy()
                hourly_snow_depth = hourly.Variables(8).ValuesAsNumpy()
                hourly_weathercode = hourly.Variables(9).ValuesAsNumpy()
                hourly_cloudcover = hourly.Variables(10).ValuesAsNumpy()
                hourly_windspeed_10m = hourly.Variables(11).ValuesAsNumpy()
                hourly_winddirection_10m = hourly.Variables(12).ValuesAsNumpy()
                hourly_shortwave_radiation = hourly.Variables(13).ValuesAsNumpy()
                hourly_direct_radiation = hourly.Variables(14).ValuesAsNumpy()
                hourly_surface_pressure = hourly.Variables(15).ValuesAsNumpy()
                hourly_visibility = hourly.Variables(16).ValuesAsNumpy()

                utc_time = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )

                local_tz = pytz.timezone(response.Timezone())
                local_time = utc_time.tz_convert(local_tz).tz_localize(None)

                hourly_data = {
                    "date": local_time,
                    "temperature_2m": hourly_temperature_2m,
                    "dew_point_2m": hourly_dew_point_2m,
                    "rain": hourly_rain,
                    "snowfall": hourly_snowfall,
                    "apparent_temperature": hourly_apparent_temperature,
                    "relative_humidity_2m": hourly_relative_humidity_2m,
                    "precipitation": hourly_precipitation,
                    "showers": hourly_showers,
                    "snow_depth": hourly_snow_depth,
                    "weathercode": hourly_weathercode,
                    "cloudcover": hourly_cloudcover,
                    "windspeed_10m": hourly_windspeed_10m,
                    "winddirection_10m": hourly_winddirection_10m,
                    "shortwave_radiation": hourly_shortwave_radiation,
                    "surface_pressure": hourly_surface_pressure,
                    "visibility": hourly_visibility
                }

                hourly_dataframe = pd.DataFrame(data=hourly_data)
                difi['ДатаЧас'] = pd.to_datetime(difi['ДатаЧас'])
                hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
                azk_weathered = pd.merge(
                    difi.rename(columns={"ДатаЧас": "datetime"}),
                    hourly_dataframe.rename(columns={"date": "datetime"}),
                    on="datetime",
                    how="inner"
                )
                weathered_dfs_list.append(azk_weathered)
                azk_weathered.to_csv(f"data/clean/azk/outages/{difi['Унікод АЗС'].values[0]}.csv", index=False)
                pbar.update(1)
            # break
    df_weathered = pd.concat(weathered_dfs_list, axis=0)
    df_weathered.to_csv('data/clean/weathered_outages.csv', index=False)


import pandas as pd
import os


def split_csv_into_three_parts(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    chunk_size = len(df) // 3
    remainder = len(df) % 3

    splits = [
        df.iloc[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)]
        for i in range(3)
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_files = []
    for i, split in enumerate(splits):
        file_path = os.path.join(output_dir, f"part_{i + 1}.csv")
        split.to_csv(file_path, index=False)
        output_files.append(file_path)

    return output_files


def combine_csv_parts(file_paths=['data/clean/weathered/part_1.csv', 'data/clean/weathered/part_2.csv', 'data/clean/weathered/part_3.csv']):
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def split_df_into_three_parts(df, output_dir):
    chunk_size = len(df) // 3
    remainder = len(df) % 3

    splits = [
        df.iloc[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)]
        for i in range(3)
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_files = []
    for i, split in enumerate(splits):
        file_path = os.path.join(output_dir, f"part_{i + 1}.csv")
        split.to_csv(file_path, index=False)
        output_files.append(file_path)

    return output_files


def get_season(pickup):
    month = pickup.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def get_season_number(pickup):
    month = pickup.month
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    elif month in [9, 10, 11]:
        return 4


def add_time_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['is_night'] = ((df['hour'] >= 21) | (df['hour'] < 6)).astype(int)
    df['Season'] = df['datetime'].apply(get_season)

    return df

def get_fully_featured_df(categorical = False, categories = [], bad_cols = True):
    df = combine_csv_parts(['data/clean/time_feature_engineered/part_1.csv', 'data/clean/time_feature_engineered/part_2.csv', 'data/clean/time_feature_engineered/part_3.csv'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['kWh'] = df['kWh'].astype(np.float32)
    df['Latitude'] = df['Latitude'].astype(np.float32)
    df['Longitude'] = df['Longitude'].astype(np.float32)
    df['InternalNum'] = df['InternalNum'].astype(np.int32)
    df['temperature_2m'] = df['temperature_2m'].astype(np.float32)
    df['dew_point_2m'] = df['dew_point_2m'].astype(np.float32)
    df['rain'] = df['rain'].astype(np.float32)
    df['snowfall'] = df['snowfall'].astype(np.float32)
    df['apparent_temperature'] = df['apparent_temperature'].astype(np.float16)
    df.drop(columns=['relative_humidity_2m', 'precipitation'])
    df['showers'] = df['showers'].astype(np.float16)
    df['snow_depth'] = df['snow_depth'].astype(np.float16)
    df['weathercode'] = df['weathercode'].astype(np.float32)
    df['cloudcover'] = df['cloudcover'].astype(np.float16)
    df['winddirection_10m'] = df['winddirection_10m'].astype(np.float16)
    df['shortwave_radiation'] = df['shortwave_radiation'].astype(np.float32)
    df['surface_pressure'] = df['surface_pressure'].astype(np.float16)
    df['visibility'] = df['visibility'].astype(np.float16)

    df['is_night'] = df['is_night'].astype(np.int8)
    df['day_of_week'] = df['day_of_week'].astype(np.int8)
    df['month'] = df['month'].astype(np.int8)
    df['day_of_month'] = df['day_of_month'].astype(np.int8)
    df['hour'] = df['hour'].astype(np.int8)
    df['day_of_year'] = df['day_of_year'].astype(np.int16)
    df['year'] = df['year'].astype(np.int16)
    df['Season'] = df['datetime'].apply(get_season_number)

    bad_coluns = ['precipitation', 'relative_humidity_2m',
        'showers', 'apparent_temperature', 'surface_pressure',
        'Season', 'is_night', 'visibility', 'month']
    if bad_cols:
        for col in bad_coluns:
            df = df.drop(columns=col)
            try:
                categories.remove(col)
            except:
                i = 1
    if categorical:
        if not categories:
            categories = ['is_night', 'day_of_week', 'month', 'day_of_month', 'hour', 'day_of_year', 'year', 'Season']
            if bad_cols:
                for col in bad_coluns:
                    try:
                        categories.remove(col)
                    except:
                        i = 1
        categories.append('InternalNum')
        df = get_dummies(df, categories)

    return df

def get_dummies(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df