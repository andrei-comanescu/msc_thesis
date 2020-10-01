import os
import twitter
import pickle as pkl
import numpy as np
import pandas as pd
import networkx as nx


def fetch_ids_and_create_neighbours_list(consumer_key, consumer_secret, access_token_key, access_token_secret,\
    graph_path, partial_user_ids_path):
    """
    Fetches the user ids for the graph nodes for which these are not available.
    Merges the existing list of node screen names and ids with the fetched pairs.
    Then, for each node, the id's of their corresponding neighbours are determined.
    Input:
        consumer_key, consumer_secret, access_token_key, access_token_secret : Twitter API access credentials
        graph_path                                                           : path to the gexf file
        partial_user_ids_path                                                : path to the partial user ids csv
    Return: None
    Files generated: four-columned headerless csv containing - user screen name, user id, list of neighbour user ids,
                     node degree; Saved in '../generated_csvs/{dataset_name}_user_ids.csv'
    """

    # create api handler
    api = twitter.Api(consumer_key=consumer_key,
                  consumer_secret=consumer_secret,
                  access_token_key=access_token_key,
                  access_token_secret=access_token_secret,
                  sleep_on_rate_limit=True)

    # get save path
    save_path = os.path.join("..", "generated_csvs", os.path.split(graph_path)[1].split("_")[0] + "_" + os.path.split(graph_path)[1].split("_")[1] + "_user_ids.csv")
    if not os.path.exists(os.path.join("..", "generated_csvs")):
        os.mkdir(os.path.join("..", "generated_csvs"))

    # read the graph
    graph  = nx.read_gexf(graph_path)
    nodes_ = set(graph.nodes())

    # read the partial user ids csv
    existing_ids = pd.read_csv(path_partial_user_ids, header = None)

    # get the missing ids
    searched_nodes     = list(nodes_ - set(existing_ids[0]))
    searched_nodes_ids = []

    iter = 1
    for node in searched_nodes:

        if iter % 100 == 0:
            print("Fetching id {}/{}".format(iter, len(searched_nodes)))

        iter += 1
        try:
            _user = api.GetUser(screen_name = node)
            searched_nodes_ids.append(_user.id)

        except twitter.error.TwitterError:
            searched_nodes_ids.append(np.nan)

    # merge the two sets of user ids
    original_nodes = nodes_ - set(searched_nodes)
    original_nodes = list(original_nodes)

    pd_user_ids = existing_ids[existing_ids[0].isin(original_nodes)].append(pd.DataFrame(list(zip(searched_nodes, searched_nodes_ids))), ignore_index = True)

    # get the list of neighbours and the node degree for each node
    neigh  = []
    weight = []

    iter = 1
    for entry in pd_user_ids[0]:

        if iter % 10000 == 0:
            print("Finding neighbours for node {}/{}".format(iter, len(pd_user_ids[0])))

        iter += 1

        neigh.append(list(pd_user_ids[pd_user_ids[0].isin(list(graph.neighbors(entry)))][1].dropna()))
        weight.append(len(list(pd_user_ids[pd_user_ids[0].isin(list(graph.neighbors(entry)))][1])))

    pd_user_ids[2] = neigh
    pd_user_ids[3] = weight

    # save the resulting dataframe as a csv
    pd_user_ids.to_csv(save_path, header = False, index = False)
