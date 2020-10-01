import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle as pkl


def get_node_degrees_daily(file_path, graph_path, yearly = None):
    """
    Parse the csv containing the daily interactions. Create a csv containing
    the real node degree (actual number of interactions) for each node on a
    day-by-day basis.
    Inputs:
        file_path  : path to the csv file containing the daily interactions
        graph_path : path to the base graph
        yearly     : do it only for a given year as string, default None
    Return: None
    Files generated:
        csv file : containing the real node degree, the columns are the nodes;
                   saved in ../../generated_csvs/{file_name}_node_degree.csv
                   and ../../generated_csvs/{file_name}_node_degree_t.csv for
                   the transposed csv; The yearly entries are saved as
                   ../../generated_csvs/{file_name}_node_degree_{year}.csv and
                   ../../generated_csvs/{file_name}_node_degree_t_{year}.csv
    """

    # read the csv and the graph
    print("Loading the graph and the csv file...")

    graph = nx.read_gexf(graph_path)

    edgelist = pd.read_csv(file_path, sep = ",", header = None, names = ["src", "dst", "time"])
    edgelist['weight'] = 1

    time_entries = list(edgelist['time'].unique())
    time_entries.sort()

    df = pd.DataFrame()

    if yearly == None:
        entries = time_entries

    else:
        entries = filter(lambda x: yearly in x, time_entries)

    # for each time entry aggregate the entries based on the interaction between users
    for time_entry in entries:
        print("Computing for entry {}".format(time_entry))

        # compute the graph
        temp_edgelist = edgelist[edgelist.time == time_entry]
        temp_edgelist = temp_edgelist.drop(columns = 'time')
        swap_cond     = temp_edgelist.src > temp_edgelist.dst
        temp_edgelist.loc[swap_cond, ['src', 'dst']] = temp_edgelist.loc[swap_cond, ['dst', 'src']].values
        agg_edgelist  = temp_edgelist.groupby([temp_edgelist['src'], temp_edgelist['dst']], as_index = False).aggregate('sum')

        # add a dummy weights to the overall graph
        for (i, j) in graph.edges:
            graph[i][j]['weight'] = 0.00001

        # add the entries weights to the graph
        for (i, j, k) in zip(list(temp_edgelist.src), list(temp_edgelist.dst), list(temp_edgelist.weight)):
            if i in list(graph.nodes) and j in list(graph.nodes):
                graph[i][j]['weight'] = graph[i][j]['weight'] + k

        # get the real node degrees
        edge_sum = {}
        d_edges  = nx.get_edge_attributes(graph, 'weight')
        for j in list(graph.nodes()):
            edge_sum[j] = 0
            val = 0
            for i in list(graph.edges(j)):
                try:
                    val += d_edges[i]
                except KeyError:
                    val += d_edges[(i[1], i[0])]

            edge_sum[j] = int(val)

        df = df.append(edge_sum, ignore_index = True)

    # get a save path
    if yearly == None:
        save_path   = os.path.join("..", "generated_csvs", os.path.split(file_path)[1].split(".")[0] + "_node_degree.csv")
        save_path_t = os.path.join("..", "generated_csvs", os.path.split(file_path)[1].split(".")[0] + "_node_degree_t.csv")

    else:
        save_path   = os.path.join("..", "generated_csvs", os.path.split(file_path)[1].split(".")[0] + "_node_degree_" + yearly + ".csv")
        save_path_t = os.path.join("..", "generated_csvs", os.path.split(file_path)[1].split(".")[0] + "_node_degree_t_" + yearly + ".csv")


    # save the csv
    print("Saving the csv to {}".format(save_path))
    df.to_csv(save_path, index = False)

    # save transposed csv
    print("Saving the transposed csv to {}".format(save_path_t))
    df.T.to_csv(save_path_t)


def merge_transposed_csvs(paths):
    """
    Merge a series of yearly transposed csvs containing the nodee degrees.
    Inputs:
        paths : list of strings representing the paths to the transposed csvs
    Return: None
    Files generated:
        csv file: contains the merged transposed csv node degree; saved as
                  ../../generated_csvs/{file_name}_node_degree_t.csv
    """

    pds = []
    for path in paths:
        pds.append(pd.read_csv(path, index_col = 0))

    pd_t = pd.concat(pds, axis = 1)

    # save the merged csv
    save_path = ""
    for i in os.path.split(paths[0])[1].split("_")[:-2]:
        save_path += i + "_"

    save_path += "t.csv"

    pd_t.to_csv(os.path.join("..", "generated_csvs", save_path))


def to_npy(csv_path):
    """
    Convert csv to npy; save it and a np array containing the indexes.
    Inputs:
        csv_path : path to the csv with the node degree
    Return: None
    Files generated:
        npy files : np.array containing the transposed csv converted to npy,
                    stored in .../npy/{node_degree_origin}/node_degreee.npy;
                    the indexes are stored as a np.array in
                    ../npy/{node_degree_origin}/index.npy
    """

    # get the save paths
    node_degree_origin = os.path.split(csv_path)[1].split("_")[0] + "_" + os.path.split(csv_path)[1].split("_")[1]

    if not os.path.exists(os.path.join("..", "npy", node_degree_origin)):
        os.makedirs(os.path.join("..", "npy", node_degree_origin))

    # load the csv and convert in to npy
    csv = pd.read_csv(csv_path, index_col = 0)

    # save the matrix and the index
    np.save(os.path.join("..", "npy", node_degree_origin, "index.npy"), np.array(csv.index))
    np.save(os.path.join("..", "npy", node_degree_origin, "node_degree.npy"), csv.to_numpy())


def pca_npy(npy_path):
    """
    Given a np.array containing the transposed node degrees perform PCA, save the
    eigenvectors and their eigenvectors and then store the dimensionaly reduced,
    to 800 features, matrix as an npy.
    Inputs:
        npy_path : path to the node degree npy file
    Return: None
    Files generated:
        npy files : eigenvectors stored in {npy_path_folder_dir}/eig_vec.npy;
                    eigenvalues are stored in {npy_path_folder_dir}/eig_val.npy
                    and the features obtained via PCA are saved in
                    {npy_path_folder_dir}/node_features_pca.npy
    """

    # load the noad degrees
    node_degree = np.load(npy_path)

    # get the empirical cov. matrix and perform the eigv. decomposition
    cov_matr         = np.cov(node_degree.T)
    eig_val, eig_vec = np.linalg.eig(cov_matr)

    eig_val = eig_val.real
    eig_vec = eig_vec.real

    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    # save the eigenvectors and the eigenvalues
    save_dir = os.path.split(npy_path)[0]
    np.save(os.path.join(save_dir, "eig_val.npy"), eig_val)
    np.save(os.path.join(save_dir, "eig_vec.npy"), eig_vec)

    # get the reduced node degrees
    reduced = np.matmul(node_degree, eig_vec[:, 0:800])

    # save the PCA-reduced features
    np.save(os.path.join(save_dir, "node_features_pca"), reduced)
