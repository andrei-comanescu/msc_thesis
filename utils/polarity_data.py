import os
import numpy as np
import pandas as pd
import networkx as nx


def create_polarity_csv(neighbors_csv_path, mcmc_path, user_polarities_paths):
    """
    Merge the neighbors csv with both the neighbourhood-based polarities and the
    following-based polarities.
    Input:
        neighbors_csv_path    : path to the csv containing the screen names and user ids
        mcmc_path             : path to the csv containing the MCMC polarity estimations
        user_polarities_paths : list containing the paths to the polarities csvs obtained
                                via following-based MCMC estimations
    Returns: None
    Files generated:
        csv file : ../generated_csvs/{file_name_base}_polarities_base.csv"
    """

    # merge the first two csvs
    original_csv = _merge_neigbors_csv_with_mcmc_polarities(neighbors_csv_path, mcmc_path)

    # get the second set of csvs into a single one
    _users = pd.DataFrame()
    for file in user_polarities_paths:
        _users = pd.concat((_users, pd.read_csv(file, header = None, sep = "\t")))

    _users = _users.drop_duplicates(0).drop(columns = 2).rename(columns = {0:1, 1:"Polarity Following"})

    # merge them
    results = pd.merge(original_csv, _users, on = 1, how = 'left')

    # save csv
    save_path = os.path.join("..", "generated_csvs", os.path.split(neighbors_csv_path)[1].split("_")[0] + "_" + os.path.split(neighbors_csv_path)[1].split("_")[1] + "_polarities_base.csv")
    results.reset_index(drop = True).to_csv(save_path, index = False)


def _merge_neigbors_csv_with_mcmc_polarities(neighbors_csv_path, mcmc_path):
    """
    Merge row-wise the csv containing the screen names and their corresponding ids
    with the csv containing the polarities estimated in R via MCMC.
    Input:
        neighbors_csv_path : path to the csv containing the screen names and user ids
        mcmc_path          : path to the csv containing the MCMC polarity estimations
    Returns: pandas dataframe containing the original neigbors_csv_columns and a new
             column for the estimated polarities
    """

    # read the csvs
    original_csv = pd.read_csv(neighbors_csv_path, header = None)
    mcmc_csv     = pd.read_csv(mcmc_path, sep = ' ')

    # get the polarities from the mcmc csv
    polarities = []
    for entry in mcmc_csv['results']:
        try:
            polarities.append(float(entry))
        except ValueError:
            polarities.append(np.nan)

    original_csv['Polarity Neighbours'] = polarities

    return original_csv


def make_neighbors_csv(graph_folder_path):
    """
    Generate a csv file containing the columns 'node', 'neighbors' and 'weight'
    for a given graph file.
    Inputs:
        graph_folder_path : path to the graph used to generate the csv
    Returns: None
    Files generated:
        csv file : ../generated_csvs/{graph_name}_neighbors_names.csv
    """

    # make the pandas dataframe
    graph = nx.read_gexf(graph_folder_path)

    node_table = pd.DataFrame(columns = ['node', 'neighbors', 'weight'])
    for node in graph.nodes():
        node_table = node_table.append({'node': node, 'neighbors': list(graph.neighbors(node)), 'weight': len(list(graph.neighbors(node)))}, ignore_index = True)

    # get the name of the save file
    save_file = os.path.split(graph_folder_path)[-1].split(".")[0]
    save_file = os.path.join("..", "generated_csvs", save_file + "_neighbors_names.csv")

    # save csv
    if not os.path.exists(os.path.join("..", "generated_csvs")):
        os.mkdir(os.path.join("..", "generated_csvs"))
    node_table.reset_index(drop = True).to_csv(save_file, index = False)


def merge_neighbors_and_polarity(neighbors_csv_path, polarity_path):
    """
    Merge the neighbors csv with the polarity csv. The resulting csv has the
    following columns: index (the indexes of the nodes in the Laplacian matrix),
    node name, neighbors, weight, polarity.
    Inputs:
        neighbors_csv_path : path to the neighbors csv
        polarity_path      : path to the polarity csv
    Returns: None
    Files generated:
        csv file : ../generated_csv/{neighbor_csv_file_name}_neighbors_polarity_merged.csv
    """

    # get the name of the save file
    save_file = ""
    for entry in os.path.split(neighbors_csv_path)[-1].split(".")[0].split("_")[0:-2]:
        save_file += entry + "_"

    save_file = os.path.join("..", "generated_csvs", save_file + "neighbors_polarity_merged.csv")

    # combine the csvs
    node_table = pd.read_csv(neighbors_csv_path)
    node_table = node_table.reset_index()

    polarity_scores = pd.read_csv(polarity_path)

    merged = pd.merge(node_table, polarity_scores, on = "node", how = "inner")

    # save csv
    if not os.path.exists(os.path.join("..", "generated_csvs")):
        os.mkdir(os.path.join("..", "generated_csvs"))
    merged.reset_index(drop = True).to_csv(save_file, index = False)
