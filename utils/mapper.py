import os
import numpy as np
import pandas as pd
import networkx as nx


def map_node_degree_graph(graph_path, node_degree_index_path):
    """
    Maps the graph indexes to the node degree ones. Generates two npy files, each
    containing an np.array, whose values represent the indexes of the mapped structures.
    Inputs:
        graph_path             : path to the graph file
        node_degree_index_path : path to the node degree index file
    Return: None
    Files generated:
        npy files : two npy files: graph_to_node_degree_index maps the graph
                    indexes to the node degree ones, such that,
                    graph_to_node_degree_index[graph_idx] == node_degree_index;
                    saved in ../npy/{dataset_name}/graph_to_node_degree_index.npy;
                    node_degree_to_graph_index maps the node degree to the graph
                    indexes, such that, node_degree_to_graph_index[node_degree_index] == graph_index;
                    saved in ../npy/{dataset_name}/node_degree_to_graph_index.npy.
    """

    # get save dir
    save_path = os.path.split(node_degree_index_path)[0]

    # read the files
    graph    = nx.read_gexf(graph_path)
    graph_idx = list(graph.nodes())

    node_degree_idx = list(np.load(node_degree_index_path, allow_pickle = True))

    # get the mapping from graph nodes to node degrees
    graph_to_node_degree_index = []
    for idx in graph_idx:
        graph_to_node_degree_index.append(node_degree_idx.index(idx))

    graph_to_node_degree_index = np.array(graph_to_node_degree_index)

    # save the mapping
    np.save(os.path.join(save_path, "graph_to_node_degree_index.npy"), graph_to_node_degree_index)

    # get the mapping from node degrees to graph nodes
    node_degree_to_graph_index = []
    for idx in node_degree_idx:
        node_degree_to_graph_index.append(graph_idx.index(idx))

    node_degree_to_graph_index = np.array(node_degree_to_graph_index)

    # save the mapping
    np.save(os.path.join(save_path, "node_degree_to_graph_index.npy"), node_degree_to_graph_index)


def order_predictions(graph_path, predictions_path):
    """
    Orders the predictions csv so that the entries follow the same indexing scheme
    as the graph/eigenvectors.

    Inputs:
        graph_path       : path to the graph
        predictions_path : path to the csv containing the predictions
    Return: None
    Files generated:
        csv file: containg the sorted classes saved as: ../keras/{dataset_name}/ordered_predictions.csv
    """

    # get the save path
    save_path = os.path.split(predictions_path)[0]

    # read the files
    graph       = nx.read_gexf(graph_path)
    predictions = pd.read_csv(predictions_path)

    # map the list to the dataframe
    graph_idx = list(graph.nodes())
    predictions['order'] = predictions['0'].apply(lambda value: graph_idx.index(value))

    # save the results
    predictions.sort_values(by = 'order').drop(columns = ['order']).reset_index(drop = True).to_csv(os.path.join(save_path, "ordered_predictions.csv"), index = False)
