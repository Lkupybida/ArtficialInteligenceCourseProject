from data_prep import *

if __name__ == "__main__":
    # xlsb_to_csv_wrapper()
    # clean_data_wrapper()
    # create_hourly_dataset()
    input_file = 'data/clean/dataset.csv'
    output_file = 'data/clean/hourly_uniformly.csv'
    # # result = convert_to_hourly_charging(input_file, output_file)
    # # print(result)
    convert_to_hourly_by_station(input_file)