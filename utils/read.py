import os
import json
import gzip
import numpy as np
import pandas as pd
import pickle as pkl


def read_aligned_pickled_eigenvectors(pickle_folder_path, pickle_file_path_agg, norm = "norm2", return_idx = False):
    """
    Read the aligned pickled eigenvectors from a given directory and sort them
    according to their date.
    Inputs:
        pickle_folder_path   : path to the directory containing the pickled eigenvectors
        pickle_file_path_agg : path to the pickle file containing the eigenvectors for the aggregated graph
        norm                 : string; determines if the eigenvectors were normed
        return_idx           : default False; boolean; returns a list of indexes that mark the order of the
                               files pertaining to the eigenvectors if True
    Return: list containg np.arrays, each np.array corresponding to one month worth
            of eigenvectors; if return_idx True then there is a second return argument
            in the form of the index list
    """

    # get the file names for the aligned eigenvectors in order
    pickle_files_aligned = [f for f in os.listdir(pickle_folder_path) if os.path.isfile(os.path.join(pickle_folder_path, f))]
    if norm == "norm2" or norm == "norm1":
        pickle_files_aligned = [pickle for pickle in pickle_files_aligned if len(pickle.split(".")[0].split("_")) == 4]
        pickle_files_aligned = [pickle for pickle in pickle_files_aligned if pickle.split(".")[0].split("_")[-1] == norm]

    elif norm == "NaN" or norm == "nan":
        pickle_files_aligned = [pickle for pickle in pickle_files_aligned if len(pickle.split(".")[0].split("_")) == 3]

    else:
        raise TypeError("Invalid input for 'norm' parameter.")


    pickle_idx = []
    for pickle_ in pickle_files_aligned:
        pickle_idx.append((pickle_.split(".")[0].split("_")[1]))

    ordered_pickle_files_ = []
    for i, j in sorted(zip(pickle_idx, pickle_files_aligned)):
        ordered_pickle_files_.append(j)


    # read the aligned eigenvectors
    xy_pairs_ = []
    for ordered_ in ordered_pickle_files_:
        with open(os.path.join(pickle_folder_path, ordered_), "rb") as pickle_file:
            _, _, xy_pairs, _, _, _, _, _, _, _ = pkl.load(pickle_file)
            xy_pairs_.append(xy_pairs)

    # read the aggregated aligned eigenvectors
    with open(pickle_file_path_agg, "rb") as pickle_file:
        _, _, xy_pairs, _, _, _, _, _, _, _ = pkl.load(pickle_file)
        xy_pairs_.append(xy_pairs)

    if return_idx:
        pickle_idx.sort()

        return xy_pairs_, pickle_idx

    return xy_pairs_


def read_daily_gz_files(path):
    """
    Read the gz files containing the daily twitter interactions. Parse through
    the files and format them as csv's.
    Inputs:
        path : path to the directory containing the daily tweeter interaction dumps
    Return: None
    Files generated:
        csv files: file containing the src_tweet (active component), dst_tweet
                    (passive component) and date as string (YYYY-MM-DD). Saved
                    in '../data/{file_name_from_path}.csv'
                   file containing the user_ids for the tweeter users that contain
                    them; file format: user name, user id. Saved in
                    '../data/{file_name_from_path}_partial_user_ids.csv'
    """

    # create the save paths
    if path[-1] == "/" or path[-1] == "\\":
        file_name = os.path.split(os.path.split(path)[0])[-1]

    else:
        file_name = os.path.split(path)[-1]

    save_path_interactions = os.path.join("..", "data", file_name + ".csv")
    save_path_user_ids     = os.path.join("..", "data", file_name + "_partial_user_ids.csv")

    # get the file names to be read
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # read the files one by one
    active  = []
    passive = []
    date    = []
    user_id = {}
    for file in files:

        # read a file
        _file = []
        with gzip.open(os.path.join(path, file), 'r') as f:
            for line in f:
                _file.append(line)

        # get the date format
        _date = file.split(".")[0].split("_")[0] + "-" + file.split(".")[0].split("_")[1] + "-" + file.split(".")[0].split("_")[2]

        # parse through the file
        for i in _file:
            # dealing with a json
            if str(i)[2] == "{":
                json_i = json.loads(i)
                # tweet
                if len(json_i['entities']['user_mentions']) == 0:
                    active.append(json_i['user']['screen_name'])
                    passive.append(json_i['user']['screen_name'])
                    date.append(_date)

                    user_id[json_i['user']['screen_name']] = json_i['user']['id']

                # reply, retweet or mention
                else:
                    for entity in json_i['entities']['user_mentions']:
                        active.append(json_i['user']['screen_name'])
                        passive.append(entity['screen_name'])
                        date.append(_date)

                        user_id[json_i['user']['screen_name']] = json_i['user']['id']
                        user_id[entity['screen_name']]         = entity['id']

            # dealing with a tsv-line
            else:
                # tweet
                if str(i).split("\\t")[2].find("@") == -1:
                    active.append(str(i).split("\\t")[0][2:])
                    passive.append(str(i).split("\\t")[0][2:])
                    date.append(_date)

                # reply, retweet or mention
                else:
                    for entry in str(i).split("\\t")[-3].split():
                        try:
                            float(entry)
                            actual_entries = list(filter(lambda user: user[0] == "@", str(i).split("\\t")[2].split()))

                            for _aentry in actual_entries:
                                if _aentry != '@' and _aentry != '@ ':
                                    if str(_aentry).find(".") == -1 and str(_aentry).find(":") == -1 and str(_aentry).find("#") == -1:
                                        active.append(str(i).split("\\t")[0][2:])
                                        passive.append(_aentry[1:])
                                        date.append(_date)

                        except ValueError:
                            active.append(str(i).split("\\t")[0][2:])
                            passive.append(entry)
                            date.append(_date)

    for i in np.where(np.array(active) == '')[0][::-1]:
        active.pop(i)
        passive.pop(i)

    for i in np.where(np.array(passive) == '')[0][::-1]:
        passive.pop(i)
        active.pop(i)

    # save the tweeter interactions csv
    pd.DataFrame(list(zip(active, passive, date))).to_csv(save_path_interactions, header = False, index = False)

    # save the partial user ids
    pd.DataFrame.from_dict(user_id, orient = 'index').reset_index().to_csv(save_path_user_ids, header = False, index = False)
