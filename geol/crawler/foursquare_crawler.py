import requests
import pandas as pd
import time
import pkg_resources
import os
import foursquare
from geol.geol_logger.geol_logger import logger
from geol.utils import constants


class Foursquare:

    def __init__(self, client_id="",
                 client_secret=""):

        self.client_id = client_id
        self.client_secret = client_secret

        # Load category names
        self.cat = pd.read_csv(pkg_resources.resource_filename(
            'geol', '/resources/category_tree.csv'), encoding="iso-8859-1")
        self.cat.set_index("cat", inplace=True)

    def start(self, grid, output, restart=None):

        # Initalize Foursquare client authentication
        fs_client = foursquare.Foursquare(self.client_id, self.client_secret)

        start_point = 0

        # Remove the file if it's already exists
        if restart is None:
            try:
                os.remove(output)
            except OSError:
                pass
        else:
            start_point = restart

        # Initialize timer and requests counter
        #tm = time.time()
        request_counter = 0

        # Build dataframe
        foursquare_data = pd.DataFrame(
            columns=["name", "address", "crossStreet", "categories", "checkin", "usercount"])

        logger.info("Calls to do: " + str(len(grid)-start_point))

        #  Iterate over the spatial grid cells. For each cell call Foursquare API
        for ind in range(start_point, len(grid)):

            request_counter += 1

            logger.info("# Requests " + str(request_counter))

            # Set bounding box for the request
            row = grid.iloc[ind]
            g = str(row.geometry)
            g_parse = g.split("((")[1].split("))")[0].split(", ")
            sw = g_parse[0].split(" ")  # South-West
            ne = g_parse[2].split(" ")  # North-East

            logger.info(str(
                ind) + " - " + str(sw[1]) + ", " + str(sw[0]) + ", " + str(ne[1]) + ", " + str(ne[0]))

            # ------------ start request! ---------------
            # To be changed if using https://github.com/mLewisLogic/foursquare/
            params = dict(
                client_id=self.client_id,
                client_secret=self.client_secret,
                v='20170801',
                sw=sw[1] + "," + sw[0],
                ne=ne[1] + ", " + ne[0],
                intent="browse"
            )

            url = "https://api.foursquare.com/v2/venues/search"

            logger.info(url)

            call_flag = False

            while (call_flag == False):

                try:
                    data = fs_client.venues.search(params)
                    call_flag = True
                except foursquare.RateLimitExceeded as rle:
                    # Time check
                    #ctm = time.time()
                    #difference_time = ctm - tm
                    waiting_time = 3600

                    logger.info("wait", waiting_time)

                    # Set type int and save
                    foursquare_data["checkin"] = foursquare_data["checkin"].astype(int)
                    foursquare_data["usercount"] = foursquare_data["usercount"].astype(int)

                    # append whatever data we got so far to the filesystem
                    if (os.path.isfile(output)):
                        foursquare_data.to_csv(
                            mode='a', header=False, index=False, encoding='utf-8')
                    else:
                        foursquare_data.to_csv(
                            output, encoding='utf-8', index=False)

                    # Reset POIs
                    # "Flush" the dataframe, so it can save the new batch of Points Of Interest
                    foursquare_data = pd.DataFrame(
                        columns=["name", "address", "crossStreet", "categories", "checkin", "usercount"])

                    time.sleep(waiting_time)
                    #tm = time.time()
                    #request_counter = 0
                except Exception as exc:
                    logger.error("ERROR: {0}".format(exc))


            # ----------- end request ---------------------

            tot = data['venues']
            print(len(tot))

            # Iterate over venues
            for glob in range(0, len(tot)):
                current_cat = data['venues'][glob]['categories']
                if len(current_cat) == 0:
                    continue

                checkin = data['venues'][glob]['stats']['checkinsCount']
                user = data['venues'][glob]['stats']['usersCount']
                name = data['venues'][glob]['name']
                current_loc = data['venues'][glob]['location']
                lat = current_loc['lat']
                lon = current_loc['lng']

                # Check presence of address and cross street
                if 'address' not in current_loc:
                    address = ""
                else:
                    address = current_loc['address']
                if 'crossStreet' not in current_loc:
                    crossStreet = ""
                else:
                    crossStreet = current_loc['crossStreet']

                # Get categories
                if ('pluralName' in current_cat[0]):
                    current_cat = current_cat[0]['pluralName']
                else:
                    current_cat = current_cat[0]['name']

                if current_cat not in self.cat.index:
                    continue

                cat_name = [self.cat.loc[current_cat][e] for e in self.cat.loc[current_cat].index if e.endswith(
                    'name') and self.cat.loc[current_cat][e] != "-"]

                # append date
                foursquare_data = foursquare_data.append({"name": name,
                                                          "address": address,
                                                          "crossStreet": crossStreet,
                                                          "categories": ':'.join(cat_name),
                                                          "checkin": checkin,
                                                          "usercount": user,
                                                          "latitude": lat,
                                                          "longitude": lon}, ignore_index=True)

            if (int(fs_client.rate_remaining) <= 0 and int(fs_client.rate_limit) > 0):
                # Time check
                #ctm = time.time()
                #difference_time = ctm - tm
                waiting_time = 3600

                logger.info("wait", waiting_time)

                # Set type int and save
                foursquare_data["checkin"] = foursquare_data["checkin"].astype(int)
                foursquare_data["usercount"] = foursquare_data["usercount"].astype(int)

                # append whatever data we got so far to the filesystem
                if (os.path.isfile(output)):
                    foursquare_data.to_csv(
                        mode='a', header=False, index=False, encoding='utf-8')
                else:
                    foursquare_data.to_csv(
                        output, encoding='utf-8', index=False)

                # Reset POIs
                # "Flush" the dataframe, so it can save the new batch of Points Of Interest
                foursquare_data = pd.DataFrame(
                    columns=["name", "address", "crossStreet", "categories", "checkin", "usercount"])

                time.sleep(waiting_time)
                #tm = time.time()
                #request_counter = 0

        # Set type int and save
        foursquare_data["checkin"] = foursquare_data["checkin"].astype(int)
        foursquare_data["usercount"] = foursquare_data["usercount"].astype(int)

        # Write to FileSystem...
        if(os.path.isfile(output)):
            foursquare_data.to_csv(
                mode='a', header=False, index=False, encoding='utf-8')

        else:
            foursquare_data.to_csv(output, encoding='utf-8', index=False)

        # Sanity check and removing duplicates
        logger.info("Sanity check and removing duplicates.")
        df = pd.read_csv(output)
        df.drop_duplicates(['name', 'latitude', 'longitude'], inplace=True)
        df.to_csv(output, encoding='utf-8', index=False)



