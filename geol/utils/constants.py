"""
Useful constants
"""

# Authors:  Gianni Barlacchi <gianni.barlacchi@gmail.com>
#           Michele Ferretti <mic.ferretti@gmail.com>

universal_crs = "epsg:3857"
default_crs = "epsg:4326"

# Foursquare crawler setup
max_request_per_hour = 4000
time_limit = 3480


# Urban Atlas
ITEM_COL = "ITEM2012"
CITIES = "CITIES"

urbanatlas_aggregation = {"High Density Residential": ["Discontinuous dense urban fabric (S.L. : 50% -  80%)",
                                                       "Continuous urban fabric (S.L. : > 80%)"],
                          "Medium Density Residential": ["Discontinuous medium density urban fabric (S.L. : 30% - 50%)"],
                          "Low Density Residential": ["Discontinuous low density urban fabric (S.L. : 10% - 30%)",
                                                      "Discontinuous very low density urban fabric (S.L. : < 10%)",
                                                      "Isolated structures"],
                          "Transportation & Utility": ["Airports", "Port areas", "Fast transit roads and associated land",
                                                       "Other roads and associated land", "Railways and associated land"],
                          "Agriculture": ["Arable land (annual crops)",
                                          "Permanent crops (vineyards, fruit trees, olive groves)",
                                          "Pastures", "Complex and mixed cultivation patterns", "Orchads"],
                          "Open Space & Recreation": ["Sports and leisure facilities", "Green urban areas",
                                                      "Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)"],
                          "Construction Sites": ["Construction sites", "Mineral extraction and dump sites"],
                          "Forests": ["Forest", "Herbaceous vegetation associations (natural grassland, moors...)"]
                          }
