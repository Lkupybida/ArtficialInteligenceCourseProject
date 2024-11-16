import pandas as pd

def xlsb_to_csv_wrapper():
    df = pd.read_excel('data/original/Транзакції_електрозарядки_01.2021-10.2024.xlsb', sheet_name='Base_01_2021-10_2024')
    df = start_end_time(df, 'Період зарядки')
    df = true_falsify(df, ['Зарядка через адаптер', 'Fishka', 'Успішна'])
    df = translate_cols(df)
    df = df.drop(columns='ChargingTime')
    df = add_lat_lon(df, 'InternalNum')
    df.to_csv('data/dataset.csv')

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
    print(locations)
    result = pd.merge(df, locations, on=col, how='left')
    result = result.drop(columns = ['InternalNum'])
    return result



