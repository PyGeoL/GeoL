"""
Useful constants
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

universal_crs = "3857"
default_crs = "epsg:4326"

# FOURSQUARE
CLIENT_ID_1 ="3LHNU3QCAKWWZHHJ5G14MTOL5XBHQATYDZS0UZELXHTXCUF2"
CLIENT_SECRET_1 = "DYLVKJCDAGTIP41PLBK0NGI0MJLFRKKADQGMYTG3NSHEO0L1"

CLIENT_ID_2 ="MEKOQ5OBYEN3AGBG2YGHR1IDBQABC4PFF3DJJWPNZ4ERLZ5X"
CLIENT_SECRET_2 = "FHTXRPYMG4W0GF15LERCATRUN05WXAYBGEIXSZFSOUC5DUJZ"

CLIENT_ID_3 ="HVI0XOSIHAFFV0YGRVHC0RQ2KAZUEGL1THLVE0WLRAEHFQQM"
CLIENT_SECRET_3 = "1PCDQCP3BTSK1JTDLRPL3G3ENZJBK34M550CARHPY0FJE1NS"

CLIENT_ID_4 = "5LDPHKOLC0GPBVK4GXO4PFKYT1BAFMU3NL1TIRBO3DY5Z3K3"
CLIENT_SECRET_4 = "MDQKJOMLUBSQDSNIUTYFFBUYA0IL2BYXZ0AHTCTBOL3S3RVG"

# Urban Atlas
ITEM_COL = "ITEM2012"
CITIES = "CITIES"

urbanatlas_aggregation = {"High Density Residential":["Discontinuous dense urban fabric (S.L. : 50% -  80%)",
                                                      "Continuous urban fabric (S.L. : > 80%)"],
                          "Medium Density Residential": ["Discontinuous medium density urban fabric (S.L. : 30% - 50%)"],
                          "Low Density Residential": ["Discontinuous low density urban fabric (S.L. : 10% - 30%)",
                                                     "Discontinuous very low density urban fabric (S.L. : < 10%)",
                                                      "Isolated structures"],
                          "Transportation & Utility": ["Airports", "Port areas", "Fast transit roads and associated land",
                                                       "Other roads and associated land", "Railways and associated land"],
                          "Agriculture" : ["Arable land (annual crops)",
                                           "Permanent crops (vineyards, fruit trees, olive groves)",
                                           "Pastures", "Complex and mixed cultivation patterns","Orchads"],
                          "Open Space & Recreation": ["Sports and leisure facilities", "Green urban areas",
                                                     "Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)"],
                          "Construction Sites": ["Construction sites", "Mineral extraction and dump sites"],
                          "Forests": ["Forest", "Herbaceous vegetation associations (natural grassland, moors...)"]
                          }