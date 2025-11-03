import os
import requests
import psycopg2
from io import StringIO
import pandas as pd


# configure from env params
bearer_token = os.getenv("BEARER_TOKEN", "")
data_url = os.getenv("DATA_URL", "")

def run_query(sql_query, db_name, db_user, db_password, db_port):
    """
    runs a query against the postgres DB specified.

    :param sql_query:
    :return:
    """
    # Database connection parameters
    connection = psycopg2.connect(dbname=db_name, user=db_user, password=db_password, host="localhost", port=db_port)

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

            ret_val = ret_val[0][0] if ret_val[0][0] else None
        except Exception as e:
            print("An error occurred:", e)

        finally:
            # Close the cursor and connection
            cursor.close()
            connection.close()

    return ret_val

def get_station_flood_level_data(station_id):
    sql = f"""
            SELECT json_agg(row_to_json(t))
            FROM (
                SELECT name, station_id,
                CASE WHEN nos_minor IS NOT NULL THEN (nos_minor) ELSE 0 END AS nos_minor,
                CASE WHEN nos_moderate IS NOT NULL THEN (nos_moderate) ELSE 0 END AS nos_moderate,
                CASE WHEN nos_major IS NOT NULL THEN (nos_major) ELSE 0 END AS nos_major,
                CASE WHEN nws_minor IS NOT NULL THEN (nws_minor) ELSE 0 END AS nws_minor,
                CASE WHEN nws_moderate IS NOT NULL THEN (nws_moderate) ELSE 0 END AS nws_moderate,
                CASE WHEN nws_major IS NOT NULL THEN (nws_major) ELSE 0 END AS nws_major
                FROM noaa_station_levels
                WHERE station_id={station_id}                
            ) t;
        """

    # get the station data
    station_flood_level_data = run_query(sql, "apsviz", os.getenv("APSVIZ_DB_USERNAME"), os.getenv("APSVIZ_DB_PASSWORD"), '5432')

    # return the station flood levels
    return station_flood_level_data

def get_time_series_data():
    """
    example numeric dataset (time series) gathered from the APZViz UI-Data/get_station_data endpoint

    :return:
    """
    print('Gathering station data over time from web services.')

    # create the SQL and get mocked up data from the DB.
    # WHERE station_name IN ('30001', '41013', '8410140')
    sql = """
          SELECT json_agg(row_to_json(t))
          FROM (SELECT DISTINCT station_id, station_name, location_name, lat, lon FROM drf_gauge_station WHERE drf_gauge_station.gauge_owner='NOAA/NOS' ORDER BY station_name) t; 
          """

    # get the station data
    station_data = run_query(sql, "apsviz_gauges", os.getenv("PG_USER"), os.getenv("PG_PASS"), '5435')

    print(f'{len(station_data)} stations collected.')

    # init the list of saved dataframes
    all_data = []

    # add in the bearer token
    headers = {"Authorization": f"Bearer {bearer_token}"}

    # get the info for each station from the UI data web service
    for station in station_data:
        # create the URL to the web service
        api_url = f"{data_url}{station['station_name']}&time_mark=2025-10-28T12%3A00%3A00Z&data_source=GFSFORECAST_NCSCV2.0&instance_name=NCSCv2.0_gfs&forcing_metclass=synoptic"

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

                # convert meters to feet
                # df['Observations'] = df['Observations'].apply(lambda x: x*3.2808399)

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

                if is_string_integer(station['station_name']):
                    # get the flood levels for this station
                    flood_level_data = get_station_flood_level_data(int(station['station_name']))

                    # if there were flood levels for this station
                    if flood_level_data is not None:
                        df['nos_minor'] = flood_level_data[0]['nos_minor']
                        df['nos_moderate'] = flood_level_data[0]['nos_moderate']
                        df['nos_major'] = flood_level_data[0]['nos_major']

                        df['nws_minor'] = flood_level_data[0]['nws_minor']
                        df['nws_moderate'] = flood_level_data[0]['nws_moderate']
                        df['nws_major'] = flood_level_data[0]['nws_major']

                # add this dataset to the list
                all_data.append(df)

    # save the data in a dataframe
    df_final = pd.concat(all_data)

    # save this for later
    df_final.to_csv('all_station_data.csv', index=False)

def is_string_integer(num_str):
    """
    checks to see if the string could be considered an integer

    :param num_str:
    :return:
    """
    return num_str.isnumeric() or num_str.isdigit()

if __name__ == '__main__':
    """
        entry point
    """
    get_time_series_data()

    print('Processing complete.')