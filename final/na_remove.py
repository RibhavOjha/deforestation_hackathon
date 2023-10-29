import pandas as pd
path = "C:/Users/RIBHAV OJHA/Desktop/Uni/Guh23/forest_aug.csv"
df = pd.read_csv(path)
df = df.dropna()
# df.to_csv("forest_aug_na_removed.csv", index=False)

lats_longs_weight = list(map(list, zip(df["lat"],
                          df["long"],
                          df["forest_area_percentage"]
                         )
               )
           )
print(lats_longs_weight[:5])

import folium

from folium.plugins import HeatMap

for coords in lats_longs_weight:

    map_obj = folium.Map(location = coords, zoom_start = 4)

    HeatMap(lats_longs_weight).add_to(map_obj)

