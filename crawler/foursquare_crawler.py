import requests
import pandas as pd
import time
import pkg_resources
import os


class Foursquare:

    def __init__(self, client_id="",
                 client_secret=""):

        self.client_id = client_id
        self.client_secret = client_secret

        # Load category names
        self.cat = pd.read_csv(pkg_resources.resource_filename(
            'geol', '/resources/category_tree.csv'))
        self.cat.set_index("cat", inplace=True)

    def start(self, grid, output):

        # Initialize timer and requests counter
        tm = time.time()
        request_counter = 0
        foursquare_data = pd.DataFrame(
            columns=["name", "address", "crossStreet", "categories", "checkin", "usercount"])

        for ind in range(0, len(grid)):
            request_counter = request_counter+1

            # Time checking, 100seconds before an hour or at 4990 requests until the end of the hour
            ctm = time.time()
            difference_time = ctm - tm

            print "# Requests " + str(request_counter)

            if((difference_time > 3500) or (((request_counter % 4990) == 0) & (request_counter > 0))):
                print("wait", (3600 - difference_time))

                sl = int(3600 - difference_time + 10)
                tm = 0

                # if (os.path.isfile(output)):
                #    with open(output, 'a') as f:
                #        foursquare_data.to_csv(f, header=False, index=False, encoding='utf-8')
                # else:
                foursquare_data.to_csv(output, encoding='utf-8', index=False)

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

            print(str(ind) + " - " + str(sw[1]) + ", " +
                  str(sw[0]) + ", " + str(ne[1]) + ", " + str(ne[0]))

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

            print (url)

            try:
                data = requests.get(url=url, params=params).json()
            except Exception as exc:
                print "ERROR: {0}".format(exc)

            # end request

            print(data)

            if(('code' in data['meta']) & (data['meta']['code'] == 403)):
                print("wait")
                time.sleep(7500)
                tm = time.time()

            if (('response' in data) & ('venues' in data['response'])):
                tot = data['response']['venues']

                # Iterate over venues
                for glob in range(0, len(tot)):
                    current_cat = data['response']['venues'][glob]['categories']
                    if len(current_cat) == 0:
                        continue

                    cat_name = []
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

        #sstr = output + "/foursquare_dataset.csv"
        # if(os.path.isfile(output)):
        #    with open(output, 'a') as f:
        #        foursquare_data.to_csv(f, header=False,index=False, encoding='utf-8')
        # else:
        foursquare_data.to_csv(output, encoding='utf-8', index=False)

        #del foursquare_data
        return foursquare_data