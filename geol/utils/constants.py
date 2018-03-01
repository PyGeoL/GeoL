"""
Useful constants
"""

# Authors:  Gianni Barlacchi <gianni.barlacchi@gmail.com>
#           Michele Ferretti <mic.ferretti@gmail.com>

import os

universal_crs = "3857"
default_crs = "epsg:4326"

# TODO CREATE INI FILE
CLIENT_ID_1 = "3LHNU3QCAKWWZHHJ5G14MTOL5XBHQATYDZS0UZELXHTXCUF2"
FOURSQUARE_API_TOKEN_1 = "DYLVKJCDAGTIP41PLBK0NGI0MJLFRKKADQGMYTG3NSHEO0L1"
#os.environ["FOURSQUARE_API_TOKEN_1"]

#CLIENT_ID_2 = "MEKOQ5OBYEN3AGBG2YGHR1IDBQABC4PFF3DJJWPNZ4ERLZ5X"
#FOURSQUARE_API_TOKEN_2 = os.environ["FOURSQUARE_API_TOKEN_2"]

#CLIENT_ID_3 = "HVI0XOSIHAFFV0YGRVHC0RQ2KAZUEGL1THLVE0WLRAEHFQQM"
#FOURSQUARE_API_TOKEN_3 = os.environ["FOURSQUARE_API_TOKEN_3"]

#CLIENT_ID_4 = "5LDPHKOLC0GPBVK4GXO4PFKYT1BAFMU3NL1TIRBO3DY5Z3K3"
#FOURSQUARE_API_TOKEN_4 = os.environ["FOURSQUARE_API_TOKEN_4"]

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
