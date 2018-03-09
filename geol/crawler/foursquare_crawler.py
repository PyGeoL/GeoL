import requests
import pandas as pd
import time
import pkg_resources
import os
from geol.geol_logger.geol_logger import logger
from geol.utils import constants

class Foursquare:

    def __init__(self, client_id="",
                 client_secret=""):

        self.client_id = client_id
        self.client_secret = client_secret

        # Load category names
        self.cat = pd.read_csv(pkg_resources.resource_filename('geol', '/resources/category_tree.csv'), encoding="iso-8859-1")
        self.cat.set_index("cat", inplace=True)

    def start(self, grid, output, restart=None):

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
        tm = time.time()
        request_counter = 0
        foursquare_data = pd.DataFrame(columns=["name", "address", "crossStreet", "categories", "checkin", "usercount"])

        logger.info("Calls to do: " + str(len(grid)-start_point))

        for ind in range(start_point, len(grid)):

            request_counter += 1

            # Time checking, 120seconds before an hour or at 4000 requests until the end of the hour
            ctm = time.time()
            difference_time = ctm - tm

            logger.info("# Requests " + str(request_counter))

            if((difference_time > constants.time_limit) or (((request_counter % constants.max_request_per_hour) == 0) & (request_counter > 0))):
                logger.info("wait", (3600 - difference_time))

                sl = int(3600 - difference_time + 10)
                tm = 0

                # Set type int and save
                foursquare_data["checkin"] = foursquare_data["checkin"].astype(int)
                foursquare_data["usercount"] = foursquare_data["usercount"].astype(int)

                if (os.path.isfile(output)):
                    foursquare_data.to_csv(mode='a', header=False, index=False, encoding='utf-8')
                else:
                    foursquare_data.to_csv(output, encoding='utf-8', index=False)

                # Reset pois
                foursquare_data = pd.DataFrame(columns=["name", "address", "crossStreet", "categories", "checkin", "usercount"])

                time.sleep(sl)
                tm = time.time()
                request_counter = 0

            # Set bounding box for the request
            row = grid.iloc[ind]
            g = str(row.geometry)
            g_parse = g.split("((")[1].split("))")[0].split(", ")
            sw = g_parse[0].split(" ")  # sw
            ne = g_parse[2].split(" ")  # ne
            #print("row:{} sw:{} ne:{}".format(row.name,sw,ne))

            logger.info(str(ind) + " - " + str(sw[1]) + ", " + str(sw[0]) + ", " + str(ne[1]) + ", " + str(ne[0]))

            # start request
            params = dict(
                client_id=self.client_id,
                client_secret=self.client_secret,
                v='20170801',
                sw=sw[1] + "," + sw[0],
                ne=ne[1] + ", " + ne[0],
                intent="browse"
            )

            url = "https://api.foursquare.com/v2/venues/search"
            #"?sw=" + sw[1] + "," + sw[0] + "&ne=" + ne[1] + "," + ne[0] + \
            #"&intent=browse&client_id=" + self.client_id + "&client_secret=" + self.client_secret + "&v=20170706"

            logger.info(url)

            try:

                data = requests.get(url=url, params=params).json()

            except Exception as exc:
                logger.error("ERROR: {0}".format(exc))

            # end request

            logger.debug(data)

            if(('code' in data['meta']) & (data['meta']['code'] == 403)):
                logger.info("Wait")
                time.sleep(7500)
                tm = time.time()

            if (('response' in data) & ('venues' in data['response'])):
                tot = data['response']['venues']

                # Iterate over venues
                for glob in range(0, len(tot)):
                    current_cat = data['response']['venues'][glob]['categories']
                    if len(current_cat) == 0:
                        continue

                    checkin = data['response']['venues'][glob]['stats']['checkinsCount']
                    user = data['response']['venues'][glob]['stats']['usersCount']
                    name = data['response']['venues'][glob]['name']
                    current_loc = data['response']['venues'][glob]['location']
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
                    # for ind in range(0,len(current_cat)):
                    #    cat_name = self._retrieve_cat(current_cat[ind]['name'])

                    # append date
                    foursquare_data = foursquare_data.append({"name": name,
                                                              "address": address,
                                                              "crossStreet": crossStreet,
                                                              "categories": ':'.join(cat_name),
                                                              "checkin": checkin,
                                                              "usercount": user,
                                                              "latitude": lat,
                                                              "longitude": lon}, ignore_index=True)

        # Set type int and save
        foursquare_data["checkin"] = foursquare_data["checkin"].astype(int)
        foursquare_data["usercount"] = foursquare_data["usercount"].astype(int)

        if(os.path.isfile(output)):
            foursquare_data.to_csv(mode='a', header=False,index=False, encoding='utf-8')

        else:
            foursquare_data.to_csv(output, encoding='utf-8', index=False)

        # Sanity check and removing duplicates
        logger.info("Sanity check and removing duplicates.")
        df = pd.read_csv(output)
        df.drop_duplicates(['name','latitude','longitude'], inplace=True)
        df.to_csv(output, encoding='utf-8', index=False)