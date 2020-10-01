import os
import pickle as pkl
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from scipy.sparse.linalg import eigsh


def smallest_magnitude_eigenvectors(path, eig_number = 500, use_largest_component = True, normalized = True, save_graph = True):
    """
    Compute the k magnitude wise smallest eigenvalues and their corresponding
    eigenvectors for a given interaction file.
    Inputs:
        path                  : path to the file containing the interactions
        eig_number            : number of smallest eigenvectors and eigenvalues
                                to be computed
        use_largest_component : use only the biggest connected component or link
                                all the disconnected components (if value is False)
        normalized            : True/False; determines if the computed eigenvectors
                                are from a normalized or unnormalized laplacian matrix
        save_graph            : True/False; if the graph is to be saved or not
    Return: None
    Files generated:
        pickle files : contains the eigenvalues and eigenvectors, saved in
                       ../../pickle/{interaction_file_name_given_by_the_path}_eig.pkl
                       (the given path is exemplified relative to this file);
                       pickled as [eigenvalues, eigenvectors]
        graph files  : contains the graph, saved in
                       ../../graph/{interaction_file_name_given_by_the_path}_{largest/full}.gexf
                       (the given path is exemplified relative to this file)
    """

    # get the save paths for the graph and the pickled values
    file_base_name    = os.path.split(path)[1].split('.')[0]
    graph_files_path  = os.path.join("..", "graph")
    pickle_files_path = os.path.join("..", "pickle")

    # read the data and aggregate the entries based on interactions between user pairs
    edgelist = pd.read_csv(path, sep = ",", header = None, names = ["src", "dst", "time"])
    edgelist['weight'] = 1
    edgelist  = edgelist.drop(columns = 'time')
    swap_cond = edgelist.src > edgelist.dst
    edgelist.loc[swap_cond, ['src', 'dst']] = edgelist.loc[swap_cond, ['dst', 'src']].values
    agg_edgelist = edgelist.groupby([edgelist['src'], edgelist['dst']], as_index = False).aggregate('sum')

    # create the graph
    graph = nx.from_pandas_edgelist(agg_edgelist, 'src', 'dst', edge_attr = True)

    # get the largest connected component
    if use_largest_component:
        largest_component       = max(nx.connected_components(graph), key = len)
        graph_largest_component = graph.subgraph(largest_component)

        if save_graph:
            # save the graph
            if not os.path.exists(graph_files_path):
                os.mkdir(graph_files_path)
            nx.write_gexf(graph_largest_component, path = os.path.join(graph_files_path, file_base_name + "_largest.gexf"))

        if normalized:
            # compute the normalized laplacian matrix
            laplacian_normalized = nx.normalized_laplacian_matrix(graph_largest_component, weight = 'weight')

            # compute the smallest k eigenvectors and eigenvalues
            eig_values_normalized, eig_vectors_normalized = eigsh(laplacian_normalized, k = eig_number, which = 'SM')

            # save the pickled variables
            if not os.path.exists(pickle_files_path):
                os.mkdir(pickle_files_path)
            with open(os.path.join(pickle_files_path, file_base_name + "_eig.pkl"), 'wb') as f:
                pkl.dump([eig_values_normalized, eig_vectors_normalized], f)

        else:
            # compute the unnormalized laplacian matrix
            laplacian_unnormalized = nx.laplacian_matrix(graph_largest_component, weight = 'weight')

            # compute the smallest k eigenvectors and eigenvalues
            eig_values_unnormalized, eig_vectors_unnormalized = eigsh(laplacian_unnormalized.asfptype(), k = eig_number, which = 'SM')

            # save the pickled variables
            if not os.path.exists(pickle_files_path):
                os.mkdir(pickle_files_path)
            with open(os.path.join(pickle_files_path, "unnormalized_" + file_base_name + "_eig.pkl"), 'wb') as f:
                pkl.dump([eig_values_unnormalized, eig_vectors_unnormalized], f)


def smallest_magnitude_eigenvectors_granular(file_path, graph_path, eig_number = 10, normalized = True, save_graph = True):
    """
    Compute the k magnitude wise smallest eigenvalues and their corresponding
    eigenvectors for each time entry of an iteraction file.
    Inputs:
        file_path             : path to the file containing the interactions
        graph_path            : graph of the interactions
        eig_number            : number of smallest eigenvectors and eigenvalues
                                to be computed
        normalized            : True/False; determines if the computed eigenvectors
                                are from a normalized or unnormalized laplacian matrix
        save_graph            : True/False; if the graph is to be saved or not
    Return: None
    Files generated:
        pickle files : contains the eigenvalues and eigenvectors, saved in
                       ../../pickle/{graph_file_name}/entry_{time_entry_number}.pkl
                       (the given path is exemplified relative to this file);
                       pickled as [eigenvalues, eigenvectors]
        graph files  : contains the graph, saved in
                       ../../graph/{graph_file_name}/entry_{time_entry_number}.gexf
                       (the given path is exemplified relative to this file) for
                       normalized eigenvectors and eigenvalues, and
                       ../../graph/{graph_file_name}/uentry_{time_entry_number}.gexf
                       in the case of unnormalized eigenvectors and eigenvalues
    """

    # get the save path for the graph and the pickled values
    folder_name       = os.path.split(graph_path)[1].split('.')[0]
    graph_files_path  = os.path.join("..", "graph", folder_name)
    pickle_files_path = os.path.join("..", "pickle", folder_name)

    # read the precomputed graph
    graph = nx.read_gexf(graph_path)

    # read the data
    edgelist = pd.read_csv(file_path, sep = ",", header = None, names = ["src", "dst", "time"])
    edgelist['weight'] = 1

    # determine the distinct time entries
    time_entries = list(edgelist['time'].unique())

    print("Time entries for {} are: {}".format(folder_name, time_entries))
    print("Computing for ...")

    # for each time entry aggregate the entries based on the interaction between users
    for time_entry in time_entries:
        print("Entry {}".format(time_entry))

        temp_edgelist = edgelist[edgelist.time == time_entry]
        temp_edgelist = temp_edgelist.drop(columns = 'time')
        swap_cond     = temp_edgelist.src > temp_edgelist.dst
        temp_edgelist.loc[swap_cond, ['src', 'dst']] = temp_edgelist.loc[swap_cond, ['dst', 'src']].values
        agg_edgelist  = temp_edgelist.groupby([temp_edgelist['src'], temp_edgelist['dst']], as_index = False).aggregate('sum')

        # add a dummy weights to the overall graph
        for (i, j) in graph.edges:
            graph[i][j]['weight'] = 0.1

        # add the entries weights to the graph
        for (i, j, k) in zip(list(temp_edgelist.src), list(temp_edgelist.dst), list(temp_edgelist.weight)):
            if i in list(graph.nodes) and j in list(graph.nodes):
                graph[i][j]['weight'] = graph[i][j]['weight'] + k

        if save_graph:
            # save the graph
            if not os.path.exists(graph_files_path):
                os.mkdir(graph_files_path)
            nx.write_gexf(graph, path = os.path.join(graph_files_path, "entry_" + str(time_entry) + ".gexf"))

        if normalized:
            # compute the normalized laplacian matrix
            laplacian_normalized = nx.normalized_laplacian_matrix(graph, weight = 'weight')

            # compute the smallest k eigenvectors and eigenvalues
            eig_values_normalized, eig_vectors_normalized = eigsh(laplacian_normalized, k = eig_number, which = 'SM')

            # save the pickled variables
            if not os.path.exists(pickle_files_path):
                os.mkdir(pickle_files_path)
            with open(os.path.join(pickle_files_path, "entry_" + str(time_entry) + ".pkl"), 'wb') as f:
                pkl.dump([eig_values_normalized, eig_vectors_normalized], f)

        else:
            # compute the unnormalized laplacian matrix
            laplacian_unnormalized = nx.laplacian_matrix(graph, weight = 'weight')

            # compute the smallest k eigenvectors and eigenvalues
            eig_values_unnormalized, eig_vectors_unnormalized = eigsh(laplacian_unnormalized.asfptype(), k = eig_number, which = 'SM')

            # save the pickled variables
            if not os.path.exists(pickle_files_path):
                os.mkdir(pickle_files_path)
            with open(os.path.join(pickle_files_path, "uentry_" + str(time_entry) + ".pkl"), 'wb') as f:
                pkl.dump([eig_values_unnormalized, eig_vectors_unnormalized], f)
