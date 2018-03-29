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

        # Build dataframe
        self.foursquare_data = pd.DataFrame(
            columns=["name", "address", "crossStreet", "categories", "checkin", "usercount"])

        # Set output
        self.output = "default.csv"

        # Initialize requests counter
        self.request_counter = 0

    def write_file(self):
        # Set type int and save
        self.foursquare_data["checkin"] = self.foursquare_data["checkin"].astype(int)
        self.foursquare_data["usercount"] = self.foursquare_data["usercount"].astype(int)

        # append whatever data we got so far to the filesystem
        if os.path.isfile(self.output):
            self.foursquare_data.to_csv(
                mode='a', header=False, index=False, encoding='utf-8')
        else:
            self.foursquare_data.to_csv(
                self.output, encoding='utf-8', index=False)

        # Reset POIs
        # "Flush" the dataframe, so it can save the new batch of Points Of Interest
        self.foursquare_data = pd.DataFrame(
            columns=["name", "address", "crossStreet", "categories", "checkin", "usercount"])


    def get_venues_search(self, fs_client, params):
        call_flag = False

        self.request_counter += 1

        logger.info("# Requests " + str(self.request_counter))

        url = "https://api.foursquare.com/v2/venues/search"

        logger.info(url)

        # ------------ start request! ---------------

        while call_flag is False:

            try:
                data = fs_client.venues.search(params)
                call_flag = True
            except foursquare.RateLimitExceeded as rle:
                waiting_time = 3600
                logger.info("wait", waiting_time)
                self.write_file()
                time.sleep(waiting_time)
            except Exception as exc:
                logger.error("ERROR: {0}".format(exc))

        # ----------- end request ---------------------

        tot = data['venues']
        logger.info("Number of venues: " + str(len(tot)))

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
            self.foursquare_data = self.foursquare_data.append({"name": name,
                                                      "address": address,
                                                      "crossStreet": crossStreet,
                                                      "categories": ':'.join(cat_name),
                                                      "checkin": checkin,
                                                      "usercount": user,
                                                      "latitude": lat,
                                                      "longitude": lon}, ignore_index=True)

        # Check if there is still rate remaining to call API
        if int(fs_client.rate_remaining) <= 0 and int(fs_client.rate_limit) > 0:
            waiting_time = 3600
            logger.info("wait", waiting_time)
            self.write_file()
            time.sleep(waiting_time)

        x1, y1 = list(map(float ,params['ne'].split(',')))
        x2, y2 = list(map(float, params['sw'].split(',')))

        # Calculate the Euclidean distance without square root
        dist_sq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

        # NOTE: dirty estimate 1 coordinate = 111,111 meters -> 10e-4 coordinate = 11.111 meters
        # Therefore, dist > 10e-4 -> dist^2 > 10e-6
        if len(tot) >= 10 & dist_sq >= 0.000001:

            x12 = (x1 + x2) / 2.0
            y12 = (y1 + y2) / 2.0

            new_params = [
                dict(ne=x12 + ", " + y1, sw=x2 + ", " + y12, intent="browse"),
                dict(ne=x1 + ", " + y1, sw=x12 + ", " + y12, intent="browse"),
                dict(ne=x12 + ", " + y12, sw=x2 + ", " + y2, intent="browse"),
                dict(ne=x1 + ", " + y12, sw=x12 + ", " + y2, intent="browse"),
            ]

            for param in new_params:
                self.get_venues_search(fs_client, param)


    def start(self, grid, output, restart=None):

        # Set output as a global variable
        self.output = output

        # Initalize Foursquare client authentication
        fs_client = foursquare.Foursquare(self.client_id, self.client_secret)

        start_point = 0

        # Remove the file if it already exists
        if restart is None:
            try:
                os.remove(output)
            except OSError:
                pass
        else:
            start_point = restart

        logger.info("Calls to do: " + str(len(grid)-start_point))

        #  Iterate over the spatial grid cells. For each cell call Foursquare API
        for ind in range(start_point, len(grid)):

            # Set bounding box for the request
            row = grid.iloc[ind]
            g = str(row.geometry)
            g_parse = g.split("((")[1].split("))")[0].split(", ")
            sw = g_parse[0].split(" ")  # South-West
            ne = g_parse[2].split(" ")  # North-East

            logger.info(str(
                ind) + " - " + str(sw[1]) + ", " + str(sw[0]) + ", " + str(ne[1]) + ", " + str(ne[0]))

            # Setup parameters for calling venue search API
            params = dict(
                sw=sw[1] + ", " + sw[0],
                ne=ne[1] + ", " + ne[0],
                intent="browse"
            )

            self.get_venues_search(fs_client, params)

        self.write_file()

        # Sanity check and removing duplicates
        logger.info("Sanity check and removing duplicates.")
        df = pd.read_csv(self.output)
        df.drop_duplicates(['name', 'latitude', 'longitude'], inplace=True)
        df.to_csv(self.output, encoding='utf-8', index=False)



