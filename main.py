from data_prep import *

if __name__ == "__main__":
    # xlsb_to_csv_wrapper()
    # clean_data_wrapper()
    # create_hourly_dataset()

    # convert_to_hourly_by_station()

    # add_weather_data()

    # add_weather_data_outages()

    df = add_time_features(optimized_weathered_df())
    split_df_into_three_parts(df, "data/clean/time_feature_engineered")