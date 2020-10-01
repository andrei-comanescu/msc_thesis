import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import pickle as pkl


def get_centers(graph_file_path = "", data_file_path = "", center_type = "top retweeters"):
    """
    Get the top k centers for a given data file while ensuring that said centers
    are part of the graph.
    Inputs:
        graph_file_path : path to the saved graph
        data_file_path  : path to the data file containing the retweets
        center_type     : determines the nature of the centers, it can take the
                          following values:
                            - "top retweeters": determines to users who retweeted the most
                            - "total retweets": determines the users who sent and received
                                                the most tweets
    Returns: the indexes of the top three centers in descending order
    """

    if center_type == "top retweeters":

        # get the top retweeters as user names
        # in the data file structured as: a b time, the retweeter is a
        edgelist = pd.read_csv(data_file_path, sep = ",", header = None, names = ["src", "dst", "time"])

        edgelist["weight"] = 1
        edgelist = edgelist.drop(columns = ["dst", "time"])

        top_retweeters = list(edgelist.groupby(edgelist['src'], as_index = False).aggregate('sum').sort_values(by = 'weight', ascending = False)['src'])

        # determine the indexes of the top three retweeters that are also present in the used graph
        graph = nx.read_gexf(graph_file_path)

        # the ordering of nodes in the Laplacian matrix, thus also in its eigenvectors is given by graph.nodes()
        node_list = list(graph.nodes())

        center_indexes = []
        retweeters_determined = 0
        for retweeter in top_retweeters:
            if retweeter in node_list:
                center_indexes.append(node_list.index(retweeter))
                retweeters_determined += 1

                if retweeters_determined >= 3:
                    break

        return center_indexes

    elif center_type == "total retweets":

        # get the total number of retweets for each user
        # that means that for each user we sum the tweets that they sends and the tweets that they receive
        graph = nx.read_gexf(graph_file_path)

        adj_mat = nx.adjacency_matrix(graph, weight = 'weight')

        values = []
        for i in range(0, adj_mat.shape[0]):
            values.append(adj_mat[i].sum())

        top_3_idx      = np.argsort(values)[-3:]
        center_indexes = list(top_3_idx[::-1])

        return center_indexes

    else:
        raise ValueError("Used center_type is not supported.")


def align_eigenvectors(pickle_file_path, c1, c2, c3, suffix = "", norm = False, norm_ord = 1, mkdir = False):
    """
    Align the smallest two eigenvectors corresponding to non-zero eigenvalues
    in accordance with centers c1, c2 and c3.
    Inputs:
        pickle_file_path : path the pickled eigenvectors and eigenvalues
        c1               : first center
        c2               : second center
        c3               : third center
        suffix           : suffix for the name of the saved pickled file: e.g., for '_centered'
                           the pickled file 'entry_22.pkl' would be named 'entry_22_aligned_centered.pkl';
                           by default it would have been named 'entry_22_aligned.pkl';
                           if the data is normed the file is labeled: 'entry_22_aligned_norm{norm_ord}.pkl' or
                           'entry_22_aligned_norm{norm_ord}_centered.pkl' for '_centered'
        norm             : True/False; if True norm the selected two eigenvectors row-wise
        norm_ord         : order of the norm
        mkdir            : make a new directory for the pickled files
    Return: None
    Files generated:
        pickle file : contains the two smallest eigenvectors and their corresponding
                      eigenvalues, the scalled eigenvectors and the scalling operations,
                      saved in ../../pickle/{pickle_file_folder_and_file}_aligned{optional_suffix}.pkl
                      (the given path is exemplified relative to this file);
                      pickled as [eigenvalues, eigenvectors, aligned_eigenvectors,
                      c1, c2, c3, x_shift_c1, y_shift_c1, rotation_c2, signature_c3]
    """

    # get the save path for the new pickled values
    if mkdir == True:
        if norm == False:
            save_pickle_path = os.path.join(os.path.split(pickle_file_path)[0] + "_aligned", os.path.split(pickle_file_path)[1].split(".")[0] + "_aligned" + suffix + ".pkl")
            if not os.path.exists(os.path.split(pickle_file_path)[0] + "_aligned"):
                os.mkdir(os.path.split(pickle_file_path)[0] + "_aligned")

        else:
            save_pickle_path = os.path.join(os.path.split(pickle_file_path)[0] + "_aligned_norm" + str(norm_ord) + suffix,\
                os.path.split(pickle_file_path)[1].split(".")[0] + "_aligned_norm" + str(norm_ord) + suffix + ".pkl")
            if not os.path.exists(os.path.split(pickle_file_path)[0] + "_aligned_norm" + str(norm_ord) + suffix):
                os.mkdir(os.path.split(pickle_file_path)[0] + "_aligned_norm" + str(norm_ord) + suffix)

    else:
        if norm == False:
            save_pickle_path = os.path.join(os.path.split(pickle_file_path)[0], os.path.split(pickle_file_path)[1].split(".")[0] + "_aligned" + suffix + ".pkl")

        else:
            save_pickle_path = os.path.join(os.path.split(pickle_file_path)[0], os.path.split(pickle_file_path)[1].split(".")[0] + "_aligned_norm" + str(norm_ord) + suffix + ".pkl")

    # load the pickled values
    with open(pickle_file_path, "rb") as pkl_file:
        eigval, eigvec = pkl.load(pkl_file)

    # align the two eigenvectors: c1 translation to origin
    xy_pairs = eigvec[:, 1:3]

    if norm == True:
        for iter in range(0, len(xy_pairs)):
            n = np.linalg.norm(xy_pairs[iter], ord = norm_ord)
            xy_pairs[iter][0] = xy_pairs[iter][0] / n
            xy_pairs[iter][1] = xy_pairs[iter][1] / n

    x_shift_c1 = xy_pairs[c1, 0]
    y_shift_c1 = xy_pairs[c1, 1]

    xy_pairs[:, 0] = xy_pairs[:, 0] - x_shift_c1
    xy_pairs[:, 1] = xy_pairs[:, 1] - y_shift_c1

    # point rotation such that c2 becomes (x_c2, 0)
    rotation_c2 = math.atan2(xy_pairs[c2, 1], xy_pairs[c2, 0])

    xx = xy_pairs[:, 0] * math.cos(rotation_c2) + xy_pairs[:, 1] * math.sin(rotation_c2)
    yy = -xy_pairs[:, 0] * math.sin(rotation_c2) + xy_pairs[:, 1] * math.cos(rotation_c2)

    xy_pairs[:, 0] = xx
    xy_pairs[:, 1] = yy

    # flip the sign of the embedding such that y_c3 > 0
    if xy_pairs[c3, 1] >= 0:
        signature_c3 = 1
    else:
        signature_c3 = -1

    xy_pairs[:, 1] = signature_c3 * xy_pairs[:, 1]

    # save the pickled variables
    with open(save_pickle_path, 'wb') as f:
        pkl.dump([eigval[1:3], eigvec[:, 1:3], xy_pairs, c1, c2, c3, x_shift_c1, y_shift_c1, rotation_c2, signature_c3], f)


def align_eigenvectors_dir(pickle_dir_path, c1, c2, c3, suffix = "", norm = False, norm_ord = 1, mkdir = False):
    """
    Align the smallest two eigenvectors corresponding to non-zero eigenvalues
    in accordance with centers c1, c2 and c3 for each pickle file that contains
    non-aligned eigenvectors (these are considered to be files of depicted as
    'entry_{number}.pkl') from a directory. If the directory contains other
    directories then it does not search them.
    Inputs:
        pickle_dir_path  : path the directory containing pickled eigenvectors and eigenvalues
        c1               : first center
        c2               : second center
        c3               : third center
        suffix           : suffix for the name of the saved pickled file: e.g., for '_centered'
                           the pickled file 'entry_22.pkl' would be named 'entry_22_aligned_centered.pkl';
                           by default it would have been named 'entry_22_aligned.pkl'
        norm             : True/False; if True norm the selected two eigenvectors row-wise
        norm_ord         : order of the norm
        mkdir            : make a new directory for the pickled files
    Return: None
    Files generated:
        pickle files : contains the two smallest eigenvectors and their corresponding
                       eigenvalues, the scalled eigenvectors and the scalling operations,
                       saved in ../../pickle/{pickle_file_folder_and_file}_aligned{optional_suffix}.pkl
                       (the given path is exemplified relative to this file);
                       pickled as [eigenvalues, eigenvectors, aligned_eigenvectors,
                       c1, c2, c3, x_shift_c1, y_shift_c1, rotation_c2]
    """

    # get all the files from the directory
    all_files = [f for f in os.listdir(pickle_dir_path) if os.path.isfile(os.path.join(pickle_dir_path, f))]

    # get only the files of the form 'entry_{number}.pkl'
    files = []
    for f in all_files:
        if len(f.split("_")) == 2:
            if f.split("_")[1].split(".")[1] == "pkl":
                files.append(f)

    # align the eigenvectors from each file
    for f in files:
        align_eigenvectors(os.path.join(pickle_dir_path, f), c1, c2, c3, suffix, norm, norm_ord, mkdir)
