import os
import requests
import psycopg2
from io import StringIO
import pandas as pd


# configure from env params
bearer_token = os.getenv("BEARER_TOKEN", "")
data_url = os.getenv("DATA_URL", "")

def run_query(sql_query):
    """
    runs a query against the APSViz gauges DB.

    :param sql_query:
    :return:
    """
    # Database connection parameters
    connection = psycopg2.connect(dbname="apsviz_gauges", user=os.getenv("PG_USER"), password=os.getenv("PG_PASS"), host="localhost", port="5435")

    # init the return
    ret_val = None

    with connection.cursor() as cursor:
        try:
            # Create a cursor object
            cursor = connection.cursor()

            # Execute an SQL query
            cursor.execute(sql_query)

            # Fetch and print results
            ret_val = cursor.fetchall()

        except Exception as e:
            print("An error occurred:", e)

        finally:
            # Close the cursor and connection
            cursor.close()
            connection.close()

    return ret_val[0][0]


def get_time_series_data():
    """
    example numeric dataset (time series) gathered from the APZViz UI-Data/get_station_data endpoint

    :return:
    """
    print('Gathering station data over time from web services.')

    # create the SQL and get mocked up data from the DB.
    # WHERE station_name IN ('30001', '41013')
    sql = """
          SELECT json_agg(row_to_json(t))
          FROM (SELECT DISTINCT station_name, location_name, lat, lon FROM drf_gauge_station ORDER BY station_name) t; 
          """

    # get the station data
    station_data = run_query(sql)

    print(f'{len(station_data)} stations collected.')

    # init the list of saved dataframes
    all_data = []

    # add in the bearer token
    headers = {"Authorization": f"Bearer {bearer_token}"}

    # get the info for each station from the UI data web service
    for station in station_data:
        # create the URL to the web service
        api_url = f"{data_url}{station['station_name']}&time_mark=2025-10-14T12%3A00%3A00Z&data_source=GFSFORECAST_NCSCV2.0&instance_name=NCSCv2.0_gfs&forcing_metclass=synoptic"

        # get the station data
        response = requests.get(api_url, headers=headers)

        # if there was any data for the station
        if response.text.strip():
            # read in the data as a CSV stream
            df = pd.read_csv(StringIO(response.text))

            # only save data that has observation data
            if 'Observations' in df.columns:
                # drop records with no observation data
                df.dropna(subset=['Observations'], inplace=True)

                for col in ['APS Nowcast','Difference (APS-OBS)','APS Forecast','NOAA Tidal Predictions']:
                    # if the column exists
                    if col in df.columns:
                        # drop unneeded column
                        df.drop(columns=[col], axis=1, inplace=True)

                # rename this column
                df.rename(columns={'time': 'datetime'}, inplace=True)

                # add cols and prefill them
                df['station'] = station['station_name']
                df['location'] = station['location_name']
                df['latitude'] = station['lat']
                df['longitude'] = station['lon']
                df['metric'] = 'water level'

                # add this dataset to the list
                all_data.append(df)

    # save the data in a dataframe
    df_final = pd.concat(all_data)

    # save this for later
    df_final.to_csv('all_station_data.csv', index=False)


if __name__ == '__main__':
    """
        entry point
    """
    get_time_series_data()

    print('Processing complete.')