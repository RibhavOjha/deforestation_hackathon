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
# Load data from Excel
df = pd.read_csv('sorted_forest_percentage.csv')
df['capital'] = ""
df["lat"] = ""
df["long"] = ""

import pandas as pd
from geopy.geocoders import Nominatim

# Sample DataFrame with a 'country_name' column
# data = {'country_name': ['United States', 'Canada', 'United Kingdom', 'France', 'Germany']}
# df = pd.DataFrame(data)

# Function to get coordinates for a country
def get_coordinates_for_country(country_name):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(country_name)
    
    if location is not None:
        latitude = location.latitude
        longitude = location.longitude
        return latitude, longitude
    else:
        return None

# Apply the function to the DataFrame to get coordinates
df['coordinates'] = df['country_name'].apply(get_coordinates_for_country)

# Extract latitude and longitude into separate columns
df[['latitude', 'longitude']] = pd.DataFrame(df['coordinates'].to_list())

# Print the DataFrame with coordinates
print(df)

sys.exit()
# def get_capitals(country):
    
#     country = CountryInfo(country)
#     capital = country.capital() 
#     return capital 



def get_lat(location):
    # Accepts: list of location names
    # returns: list of lists containing latitude and longitude coordinates for the specified locations
    
    geolocator = Nominatim(user_agent="bytescout", timeout=None)
    loc_lat_long = geolocator.geocode(query = location)

    return loc_lat_long.latitude
    
def get_long(location):
    geolocator = Nominatim(user_agent="bytescout", timeout=None)
    loc_lat_long = geolocator.geocode(query = location)

    return loc_lat_long.longitude

import pandas as pd
from geopy.geocoders import Nominatim

# Initialize the geolocator
geolocator = Nominatim(user_agent="bytescout", timeout=None)

# Create a function to geocode a location
def getloc_lats_longs(location):
    location = location.strip()  # Remove leading/trailing spaces
    try:
        loc_lat_long = geolocator.geocode(query=location)
        if loc_lat_long:
            return loc_lat_long.latitude, loc_lat_long.longitude
        else:
            return "error lat", "error long"
    except Exception as e:
        return "error lat", "error long"

# Assuming you have a DataFrame 'df' with a 'country_name' column

# Add new columns to store the results
df['capital'] = ""
df['lat'] = ""
df['long'] = ""

# Iterate over the DataFrame
for index, row in df.iterrows():
    if index == 0:
        continue

    capital = get_capital(row['country_name'])  # You need to define 'get_capital'
    lat, long = getloc_lats_longs(capital)

    # Update the DataFrame with the values
    df.at[index, 'capital'] = capital
    df.at[index, 'lat'] = lat
    df.at[index, 'long'] = long
    print("index", index)
    print("lat", lat)

# Now 'df' contains the updated values

print(df)


# df_aug = pd.DataFrame(data, columns=['capital', 'lat', 'long'])
# df.to_csv("forest_aug.csv", index=False)
    
# # Apply the get_lat and get_long functions to the "country_name" column
# df['lat'] = df["country_name"].apply(get_lat)
# df['long'] = df["country_name"].apply(get_long)

# # Create a DataFrame with the desired columns
# lats_longs_weight = df[['lat', 'long', 'forest_area_percentage']].values.tolist()

# print(lats_longs_weight)

