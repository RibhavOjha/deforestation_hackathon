
from itertools import count
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import streamlit as st
import folium as fl
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime
import wikipedia
import pandas as pd
from sklearn.model_selection import train_test_split
from countryinfo import CountryInfo
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



path = "C:/Users/RIBHAV OJHA/Desktop/Uni/Guh23/forest_percentage.csv"
df = pd.read_csv(path)
df = df.dropna()
def convert_csv_to_json(csv_file):
    outer_country_dict = {}
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Iterate through each row in the CSV file
        for row in csv_reader:
            country_code, country_name, year, _, _, _, forest_area_percentage = row
            # Check if the country is already in the data_structure
            if country_name in outer_country_dict:
                # Check if the year is already in the country's sub-dictionary
                if year in outer_country_dict[country_name]:
                    # Add the forest_area_percentage for the specific year
                    outer_country_dict[country_name][year] = forest_area_percentage
                else:
                    # Create a new year entry for the country
                    outer_country_dict[country_name][year] = forest_area_percentage
            else:
                # Create a new entry for the country and the year
                outer_country_dict[country_name] = {year: forest_area_percentage}
    return outer_country_dict

def convert_json_to_csv(json):
    data = []
    seen = set()
    for country, values in json.items():
        for year, forest_area in values.items():
            print(country, year)
            if country not in seen: 
                if country == "country_name": 
                    continue
                lat, long = get_coordinates_for_country(country)
                data.append([country, year, forest_area, lat, long])
                seen.add(country)

            else: 
                data.append([country, year, forest_area, None, None])

    df_sorted = pd.DataFrame(data, columns=['country_name', 'year', 'forest_area_percentage', 'lat', 'long'])
    df_sorted.to_csv("forest_aug.csv", index=False)

# def get_capital(country):
#     try: 
#         country = CountryInfo(country)
#         capital = country.capital() 
#         return capital 
#     except: pass

def get_coordinates_for_country(country_name):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(country_name)

    if location is not None:
        latitude = location.latitude
        longitude = location.longitude
        return latitude, longitude
    else:
        return "none", "none"
    
def getloc_lats_longs(location):
    # Accepts: list of location names
    # returns: list of lists containing latitude and longitude coordinates for the specified locations
    
    geolocator = Nominatim(user_agent="bytescout", timeout=None)
    loc_lat_long = geolocator.geocode(query = location)

    return(loc_lat_long.latitude, loc_lat_long.longitude)
dict_data = convert_csv_to_json(path)
# print(dict_data)
convert_json_to_csv(dict_data)


# '''' {"india": {"2017" : "forest_area"}, "2012": forset_area}'''


