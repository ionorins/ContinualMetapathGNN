import torch
import random as rd
from os.path import join
from os.path import isfile
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import tqdm
import pickle
from secrets import token_hex

from .dataset import Dataset
from torch_geometric.data import download_url, extract_zip
from ..parser import parse_ml25m, parse_mlsmall


def save_df(df, path):
    df.to_csv(path, sep=";", index=False)


def reindex_df_mlsmall(movies, ratings, tagging):
    """

    Args:
        movies:
        ratings:
        tagging:
        genome_tagging:
        genome_tags:

    Returns:

    """
    # Reindex uid
    unique_uids = np.sort(ratings.uid.unique()).astype(np.int)
    uids = np.arange(unique_uids.shape[0]).astype(np.int)
    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(unique_uids, uids)}
    ratings["uid"] = np.array(
        [raw_uid2uid[raw_uid] for raw_uid in ratings.uid], dtype=np.int
    )
    tagging["uid"] = np.array(
        [raw_uid2uid[raw_uid] for raw_uid in tagging.uid], dtype=np.int
    )

    # Reindex iid
    unique_iids = np.sort(movies.iid.unique()).astype(np.int)
    iids = np.arange(unique_iids.shape[0]).astype(np.int)
    raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(unique_iids, iids)}
    movies["iid"] = np.array(
        [raw_iid2iid[raw_iid] for raw_iid in movies.iid], dtype=np.int
    )
    ratings["iid"] = np.array(
        [raw_iid2iid[raw_iid] for raw_iid in ratings.iid], dtype=np.int
    )
    tagging["iid"] = np.array(
        [raw_iid2iid[raw_iid] for raw_iid in tagging.iid], dtype=np.int
    )

    # Create tid
    unique_tags = np.sort(tagging.tag.unique()).astype(np.str)
    tids = np.arange(unique_tags.shape[0]).astype(np.int)
    tags = pd.DataFrame({"tid": tids, "tag": unique_tags})
    tag2tid = {tag: tid for tag, tid in zip(unique_tags, tids)}
    tagging["tid"] = np.array([tag2tid[tag] for tag in tagging.tag], dtype=np.int)
    tagging = tagging.drop(columns=["tag"])

    return movies, ratings, tagging, tags


def reindex_df_ml25m(movies, ratings, tagging, genome_tagging, genome_tags):
    """

    Args:
        movies:
        ratings:
        tagging:
        genome_tagging:
        genome_tags:

    Returns:

    """
    # Reindex uid
    unique_uids = np.sort(ratings.uid.unique()).astype(np.int)
    uids = np.arange(unique_uids.shape[0]).astype(np.int)
    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(unique_uids, uids)}
    ratings["uid"] = np.array(
        [raw_uid2uid[raw_uid] for raw_uid in ratings.uid], dtype=np.int
    )
    tagging["uid"] = np.array(
        [raw_uid2uid[raw_uid] for raw_uid in tagging.uid], dtype=np.int
    )

    # Reindex iid
    unique_iids = np.sort(movies.iid.unique()).astype(np.int)
    iids = np.arange(unique_iids.shape[0]).astype(np.int)
    raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(unique_iids, iids)}
    movies["iid"] = np.array(
        [raw_iid2iid[raw_iid] for raw_iid in movies.iid], dtype=np.int
    )
    ratings["iid"] = np.array(
        [raw_iid2iid[raw_iid] for raw_iid in ratings.iid], dtype=np.int
    )
    tagging["iid"] = np.array(
        [raw_iid2iid[raw_iid] for raw_iid in tagging.iid], dtype=np.int
    )
    genome_tagging["iid"] = np.array(
        [raw_iid2iid[raw_iid] for raw_iid in genome_tagging.iid], dtype=np.int
    )

    # Create tid
    unique_tags = np.sort(tagging.tag.unique()).astype(np.str)
    tids = np.arange(unique_tags.shape[0]).astype(np.int)
    tags = pd.DataFrame({"tid": tids, "tag": unique_tags})
    tag2tid = {tag: tid for tag, tid in zip(unique_tags, tids)}
    tagging["tid"] = np.array([tag2tid[tag] for tag in tagging.tag], dtype=np.int)
    tagging = tagging.drop(columns=["tag"])

    # Reindex genome_tid
    unique_genome_tids = np.sort(genome_tags.genome_tid.unique()).astype(np.int)
    genome_tids = np.arange(unique_genome_tids.shape[0]).astype(np.int)
    raw_genome_tid2genome_tid = {
        raw_genome_tid: genome_tid
        for raw_genome_tid, genome_tid in zip(unique_genome_tids, genome_tids)
    }
    genome_tags["genome_tid"] = np.array(
        [
            raw_genome_tid2genome_tid[raw_genome_tid]
            for raw_genome_tid in genome_tags.genome_tid
        ],
        dtype=np.int,
    )
    genome_tagging["genome_tid"] = np.array(
        [
            raw_genome_tid2genome_tid[raw_genome_tid]
            for raw_genome_tid in genome_tagging.genome_tid
        ]
    )

    return movies, ratings, tagging, tags, genome_tagging, genome_tags


def drop_infrequent_concept_from_str(df, concept_name, num_occs):
    concept_strs = [concept_str for concept_str in df[concept_name]]
    duplicated_concept = [concept_str.split(",") for concept_str in concept_strs]
    duplicated_concept = list(itertools.chain.from_iterable(duplicated_concept))
    writer_counter_dict = Counter(duplicated_concept)
    del writer_counter_dict[""]
    del writer_counter_dict["N/A"]
    unique_concept = [k for k, v in writer_counter_dict.items() if v >= num_occs]
    concept_strs = [
        ",".join(
            [concept for concept in concept_str.split(",") if concept in unique_concept]
        )
        for concept_str in concept_strs
    ]
    df[concept_name] = concept_strs
    return df


def generate_mlsmall_hete_graph(
    movies,
    ratings,
    tagging,
    start,
    stop,
    init_time,
    single,
    future_testing,
    last_task_accuracy,
    testing_edges,
):
    def get_concept_num_from_str(df, concept_name):
        concept_strs = [concept_str.split(",") for concept_str in df[concept_name]]
        concepts = set(itertools.chain.from_iterable(concept_strs))
        concepts.remove("")
        num_concepts = len(concepts)
        return list(concepts), num_concepts

    #########################  Define entities  #########################
    unique_uids = list(np.sort(ratings.uid.unique()))
    num_uids = len(unique_uids)

    unique_iids = list(np.sort(ratings.iid.unique()))
    num_iids = len(unique_iids)

    future_ratings = ratings[ratings.timestamp >= stop]
    ratings = ratings[ratings.timestamp < stop]

    unique_genres = list(movies.keys()[3:22])
    num_genres = len(unique_genres)

    unique_years = list(movies.year.unique())
    num_years = len(unique_years)

    unique_directors, num_directors = get_concept_num_from_str(movies, "directors")
    unique_actors, num_actors = get_concept_num_from_str(movies, "actors")
    unique_writers, num_writers = get_concept_num_from_str(movies, "writers")

    unique_tids = list(np.sort(tagging.tid.unique()))
    num_tids = len(unique_tids)

    dataset_property_dict = {}
    dataset_property_dict["unique_uids"] = unique_uids
    dataset_property_dict["num_uids"] = num_uids
    dataset_property_dict["unique_iids"] = unique_iids
    dataset_property_dict["num_iids"] = num_iids
    dataset_property_dict["unique_genres"] = unique_genres
    dataset_property_dict["num_genres"] = num_genres
    dataset_property_dict["unique_years"] = unique_years
    dataset_property_dict["num_years"] = num_years
    dataset_property_dict["unique_directors"] = unique_directors
    dataset_property_dict["num_directors"] = num_directors
    dataset_property_dict["unique_actors"] = unique_actors
    dataset_property_dict["num_actors"] = num_actors
    dataset_property_dict["unique_writers"] = unique_writers
    dataset_property_dict["num_writers"] = num_writers
    dataset_property_dict["unique_tids"] = unique_tids
    dataset_property_dict["num_tids"] = num_tids

    #########################  Define number of entities  #########################
    num_nodes = (
        num_uids
        + num_iids
        + num_genres
        + num_years
        + num_directors
        + num_actors
        + num_writers
        + num_tids
    )
    num_node_types = 8
    dataset_property_dict["num_nodes"] = num_nodes
    dataset_property_dict["num_node_types"] = num_node_types
    types = ["uid", "iid", "genre", "year", "director", "actor", "writer", "tid"]
    num_nodes_dict = {
        "uid": num_uids,
        "iid": num_iids,
        "genre": num_genres,
        "year": num_years,
        "director": num_directors,
        "actor": num_actors,
        "writer": num_writers,
        "tid": num_tids,
    }

    #########################  Define entities to node id map  #########################
    type_accs = {}
    nid2e_dict = {}
    acc = 0
    type_accs["uid"] = acc
    uid2nid = {uid: i + acc for i, uid in enumerate(unique_uids)}
    for i, uid in enumerate(unique_uids):
        nid2e_dict[i + acc] = ("uid", uid)
    acc += num_uids
    type_accs["iid"] = acc
    iid2nid = {iid: i + acc for i, iid in enumerate(unique_iids)}
    for i, iid in enumerate(unique_iids):
        nid2e_dict[i + acc] = ("iid", iid)
    acc += num_iids
    type_accs["genre"] = acc
    genre2nid = {genre: i + acc for i, genre in enumerate(unique_genres)}
    for i, genre in enumerate(unique_genres):
        nid2e_dict[i + acc] = ("genre", genre)
    acc += num_genres
    type_accs["year"] = acc
    year2nid = {year: i + acc for i, year in enumerate(unique_years)}
    for i, year in enumerate(unique_years):
        nid2e_dict[i + acc] = ("year", year)
    acc += num_years
    type_accs["director"] = acc
    director2nid = {director: i + acc for i, director in enumerate(unique_directors)}
    for i, director in enumerate(unique_directors):
        nid2e_dict[i + acc] = ("director", director)
    acc += num_directors
    type_accs["actor"] = acc
    actor2nid = {actor: i + acc for i, actor in enumerate(unique_actors)}
    for i, actor in enumerate(unique_actors):
        nid2e_dict[i + acc] = ("actor", actor)
    acc += num_actors
    type_accs["writer"] = acc
    writer2nid = {writer: i + acc for i, writer in enumerate(unique_writers)}
    for i, writer in enumerate(unique_writers):
        nid2e_dict[i + acc] = ("writer", writer)
    acc += num_writers
    type_accs["tid"] = acc
    tag2nid = {tid: i + acc for i, tid in enumerate(unique_tids)}
    for i, tid in enumerate(unique_tids):
        nid2e_dict[i + acc] = ("tid", tid)
    e2nid_dict = {
        "uid": uid2nid,
        "iid": iid2nid,
        "genre": genre2nid,
        "year": year2nid,
        "director": director2nid,
        "actor": actor2nid,
        "writer": writer2nid,
        "tid": tag2nid,
    }
    dataset_property_dict["e2nid_dict"] = e2nid_dict
    dataset_property_dict["nid2e_dict"] = nid2e_dict

    #########################  create graphs  #########################
    edge_index_nps = {}
    print("Creating item attribute edges...")
    inids = [e2nid_dict["iid"][iid] for iid in movies.iid]
    year_nids = [e2nid_dict["year"][year] for year in movies.year]
    year2item_edge_index_np = np.vstack((np.array(year_nids), np.array(inids)))

    genre_nids = []
    inids = []
    for genre in unique_genres:
        iids = movies[movies[genre]].iid
        inids += [e2nid_dict["iid"][iid] for iid in iids]
        genre_nids += [e2nid_dict["genre"][genre] for _ in range(iids.shape[0])]
    genre2item_edge_index_np = np.vstack((np.array(genre_nids), np.array(inids)))

    inids = [e2nid_dict["iid"][iid] for iid in movies.iid]
    directors_list = [
        [director for director in directors.split(",") if director != ""]
        for directors in movies.directors
    ]
    directors_nids = [
        [e2nid_dict["director"][director] for director in directors]
        for directors in directors_list
    ]
    directors_nids = list(itertools.chain.from_iterable(directors_nids))
    d_inids = [
        [i_nid for _ in range(len(directors_list[idx]))]
        for idx, i_nid in enumerate(inids)
    ]
    d_inids = list(itertools.chain.from_iterable(d_inids))
    director2item_edge_index_np = np.vstack(
        (np.array(directors_nids), np.array(d_inids))
    )

    actors_list = [
        [actor for actor in actors.split(",") if actor != ""]
        for actors in movies.actors
    ]
    actor_nids = [
        [e2nid_dict["actor"][actor] for actor in actors] for actors in actors_list
    ]
    actor_nids = list(itertools.chain.from_iterable(actor_nids))
    a_inids = [
        [i_nid for _ in range(len(actors_list[idx]))] for idx, i_nid in enumerate(inids)
    ]
    a_inids = list(itertools.chain.from_iterable(a_inids))
    actor2item_edge_index_np = np.vstack((np.array(actor_nids), np.array(a_inids)))

    writers_list = [
        [writer for writer in writers.split(",") if writer != ""]
        for writers in movies.writers
    ]
    writer_nids = [
        [e2nid_dict["writer"][writer] for writer in writers] for writers in writers_list
    ]
    writer_nids = list(itertools.chain.from_iterable(writer_nids))
    w_inids = [
        [i_nid for _ in range(len(writers_list[idx]))]
        for idx, i_nid in enumerate(inids)
    ]
    w_inids = list(itertools.chain.from_iterable(w_inids))
    writer2item_edge_index_np = np.vstack((np.array(writer_nids), np.array(w_inids)))
    edge_index_nps["year2item"] = year2item_edge_index_np
    edge_index_nps["genre2item"] = genre2item_edge_index_np
    edge_index_nps["director2item"] = director2item_edge_index_np
    edge_index_nps["actor2item"] = actor2item_edge_index_np
    edge_index_nps["writer2item"] = writer2item_edge_index_np

    unids = [e2nid_dict["uid"][uid] for uid in tagging.uid]
    tnids = [e2nid_dict["tid"][tid] for tid in tagging.tid]
    inids = [e2nid_dict["iid"][iid] for iid in tagging.iid]
    tag2user_edge_index_np = np.vstack((np.array(tnids), np.array(unids)))
    tag2item_edge_index_np = np.vstack((np.array(tnids), np.array(inids)))
    edge_index_nps["tag2user"] = tag2user_edge_index_np
    edge_index_nps["tag2item"] = tag2item_edge_index_np

    print("Creating rating property edges...")
    test_pos_unid_inid_map, neg_unid_inid_map = {}, {}

    rating_np = np.zeros((0,))
    user2item_edge_index_np = np.zeros((2, 0))

    # if future_testing:
    #     test_sorted_ratings = ratings[ratings.timestamp >= start]
    unfiltered_sorted_ratings = ratings.sort_values("uid")

    unique_uids = list(np.sort(ratings.uid.unique()))
    unique_iids = list(np.sort(ratings.iid.unique()))

    # filter out unseed iids from future_ratings
    future_ratings = future_ratings[future_ratings.iid.isin(unique_iids)]

    # if online or single only keep ratings from current timeframe
    if single:
        ratings = ratings[ratings.timestamp >= start]
    else:
        ratings = ratings[ratings.timestamp >= init_time]

    sorted_ratings = ratings.sort_values("uid")
    edge_values = {}

    pbar = tqdm.tqdm(unique_uids, total=len(unique_uids))
    for uid in pbar:
        pbar.set_description("Creating the edges for the user {}".format(uid))
        uid_ratings = sorted_ratings[sorted_ratings.uid == uid].sort_values("timestamp")
        uid_iids = uid_ratings.iid.to_numpy()
        uid_edge_values = uid_ratings.rating.to_numpy()
        uid_timestamps = uid_ratings.timestamp.to_numpy()

        unfiltered_uid_ratings = unfiltered_sorted_ratings[
            unfiltered_sorted_ratings.uid == uid
        ].sort_values("timestamp")
        unfiltered_uid_iids = unfiltered_uid_ratings.iid.to_numpy()
        unfiltered_uid_ratings = unfiltered_uid_ratings.rating.to_numpy()

        unid = e2nid_dict["uid"][uid]

        train_pos_uid_iids = list(uid_iids)
        train_pos_uid_ratings = uid_edge_values
        test_pos_uid_inids = []

        # uid_test_ratings = future_ratings[future_ratings.uid == uid].sort_values('timestamp')
        # uid_test_iids = uid_test_ratings.iid.to_numpy()
        # uid_test_ratings = uid_test_ratings.rating.to_numpy()
        # test_pos_uid_iids = list(uid_test_iids)[:4]
        # test_pos_uid_ratings = uid_test_ratings[:4]
        # test_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in test_pos_uid_iids]

        # for nid, rating in zip(test_pos_uid_inids, test_pos_uid_ratings):
        #     edge_values[(unid, nid)] = rating / 5

        if not future_testing:
            if len(train_pos_uid_iids) == 0:
                continue
            last_rating_idx = len(uid_edge_values) - 1
            for i in range(last_rating_idx, -1, -1):
                if uid_edge_values[i] >= 0 and (uid_timestamps[i] >= start or not last_task_accuracy):
                    edge_values[(unid, e2nid_dict["iid"][train_pos_uid_iids[i]])] = (
                        uid_edge_values[i] / 5
                    )
                    test_pos_uid_iids = [train_pos_uid_iids[i]]
                    if uid not in testing_edges:
                        testing_edges[uid] = []
                    testing_edges[uid].append(train_pos_uid_iids[i])
                    test_pos_uid_inids = [
                        e2nid_dict["iid"][iid] for iid in test_pos_uid_iids
                    ]
                    train_pos_uid_iids.pop(i)
                    uid_edge_values = np.delete(uid_edge_values, i)
                    break

        # remove testing edges from training edges
        train_pos_uid_iids = list(
            set(train_pos_uid_iids) - set(testing_edges.get(uid, []))
        )
        train_pos_uid_inids = [e2nid_dict["iid"][iid] for iid in train_pos_uid_iids]
        neg_uid_iids = list(set(unique_iids) - set(unfiltered_uid_iids))
        neg_uid_inids = [e2nid_dict["iid"][iid] for iid in neg_uid_iids]

        if len(test_pos_uid_inids) > 0:
            test_pos_unid_inid_map[unid] = test_pos_uid_inids

        neg_unid_inid_map[unid] = neg_uid_inids

        unid_user2item_edge_index_np = np.array(
            [[unid for _ in range(len(train_pos_uid_inids))], train_pos_uid_inids]
        )
        user2item_edge_index_np = np.hstack(
            [user2item_edge_index_np, unid_user2item_edge_index_np]
        )

        rating_np = np.concatenate([rating_np, train_pos_uid_ratings])

    print(
        f"Total testing inids: {sum([len(v) for v in test_pos_unid_inid_map.values()])}"
    )
    dataset_property_dict["rating_np"] = rating_np
    edge_index_nps["user2item"] = user2item_edge_index_np

    dataset_property_dict["edge_index_nps"] = edge_index_nps
    (
        dataset_property_dict["test_pos_unid_inid_map"],
        dataset_property_dict["neg_unid_inid_map"],
    ) = (test_pos_unid_inid_map, neg_unid_inid_map)

    print("Building edge type map...")
    edge_type_dict = {
        edge_type: edge_type_idx
        for edge_type_idx, edge_type in enumerate(list(edge_index_nps.keys()))
    }
    dataset_property_dict["edge_type_dict"] = edge_type_dict
    dataset_property_dict["num_edge_types"] = len(list(edge_index_nps.keys()))

    print("Building the item occurrence map...")
    item_count = ratings["iid"].value_counts()
    item_nid_occs = {}
    for iid in unique_iids:
        item_nid_occs[e2nid_dict["iid"][iid]] = item_count.get(iid, 0)

    dataset_property_dict["item_nid_occs"] = item_nid_occs

    # New functionality for pytorch geometric like dataset
    dataset_property_dict["types"] = types
    dataset_property_dict["num_nodes_dict"] = num_nodes_dict
    dataset_property_dict["type_accs"] = type_accs

    user2item = dataset_property_dict["edge_index_nps"]["user2item"]
    ratings = dataset_property_dict["rating_np"]

    edge_values.update(
        {
            (int(user2item[0][i]), int(user2item[1][i])): ratings[i] / 5
            for i in range(len(user2item[0]))
        }
    )

    dataset_property_dict["edge_values"] = edge_values

    return dataset_property_dict


def generate_ml25m_hete_graph(movies, ratings, tagging, genome_tagging):
    def get_concept_num_from_str(df, concept_name):
        concept_strs = [concept_str.split(",") for concept_str in df[concept_name]]
        concepts = set(itertools.chain.from_iterable(concept_strs))
        concepts.remove("")
        num_concepts = len(concepts)
        return list(concepts), num_concepts

    #########################  Define entities  #########################
    unique_uids = list(np.sort(ratings.uid.unique()))
    num_uids = len(unique_uids)

    unique_iids = list(np.sort(ratings.iid.unique()))
    num_iids = len(unique_iids)

    unique_genres = list(movies.keys()[3:23])
    num_genres = len(unique_genres)

    unique_years = list(movies.year.unique())
    num_years = len(unique_years)

    unique_directors, num_directors = get_concept_num_from_str(movies, "directors")
    unique_actors, num_actors = get_concept_num_from_str(movies, "actors")
    unique_writers, num_writers = get_concept_num_from_str(movies, "writers")

    unique_tids = list(np.sort(tagging.tid.unique()))
    num_tids = len(unique_tids)

    unique_genome_tids = list(np.sort(genome_tagging.genome_tid.unique()))
    num_genome_tids = len(unique_genome_tids)

    dataset_property_dict = {}
    dataset_property_dict["unique_uids"] = unique_uids
    dataset_property_dict["num_uids"] = num_uids
    dataset_property_dict["unique_iids"] = unique_iids
    dataset_property_dict["num_iids"] = num_iids
    dataset_property_dict["unique_genres"] = unique_genres
    dataset_property_dict["num_genres"] = num_genres
    dataset_property_dict["unique_years"] = unique_years
    dataset_property_dict["num_years"] = num_years
    dataset_property_dict["unique_directors"] = unique_directors
    dataset_property_dict["num_directors"] = num_directors
    dataset_property_dict["unique_actors"] = unique_actors
    dataset_property_dict["num_actors"] = num_actors
    dataset_property_dict["unique_writers"] = unique_writers
    dataset_property_dict["num_writers"] = num_writers
    dataset_property_dict["unique_tids"] = unique_tids
    dataset_property_dict["num_tids"] = num_tids
    dataset_property_dict["unique_genome_tids"] = unique_genome_tids
    dataset_property_dict["num_genome_tids"] = num_genome_tids

    #########################  Define number of entities  #########################
    num_nodes = (
        num_uids
        + num_iids
        + num_genres
        + num_years
        + num_directors
        + num_actors
        + num_writers
        + num_tids
        + num_genome_tids
    )
    num_node_types = 9
    dataset_property_dict["num_nodes"] = num_nodes
    dataset_property_dict["num_node_types"] = num_node_types
    types = [
        "uid",
        "iid",
        "genre",
        "year",
        "director",
        "actor",
        "writer",
        "tid",
        "genome_tid",
    ]
    num_nodes_dict = {
        "uid": num_uids,
        "iid": num_iids,
        "genre": num_genres,
        "year": num_years,
        "director": num_directors,
        "actor": num_actors,
        "writer": num_writers,
        "tid": num_tids,
        "genome_tid": num_genome_tids,
    }

    #########################  Define entities to node id map  #########################
    type_accs = {}
    nid2e_dict = {}
    acc = 0
    type_accs["uid"] = acc
    uid2nid = {uid: i + acc for i, uid in enumerate(unique_uids)}
    for i, uid in enumerate(unique_uids):
        nid2e_dict[i + acc] = ("uid", uid)
    acc += num_uids
    type_accs["iid"] = acc
    iid2nid = {iid: i + acc for i, iid in enumerate(unique_iids)}
    for i, iid in enumerate(unique_iids):
        nid2e_dict[i + acc] = ("iid", iid)
    acc += num_iids
    type_accs["genre"] = acc
    genre2nid = {genre: i + acc for i, genre in enumerate(unique_genres)}
    for i, genre in enumerate(unique_genres):
        nid2e_dict[i + acc] = ("genre", genre)
    acc += num_genres
    type_accs["year"] = acc
    year2nid = {year: i + acc for i, year in enumerate(unique_years)}
    for i, year in enumerate(unique_years):
        nid2e_dict[i + acc] = ("year", year)
    acc += num_years
    type_accs["director"] = acc
    director2nid = {director: i + acc for i, director in enumerate(unique_directors)}
    for i, director in enumerate(unique_directors):
        nid2e_dict[i + acc] = ("director", director)
    acc += num_directors
    type_accs["actor"] = acc
    actor2nid = {actor: i + acc for i, actor in enumerate(unique_actors)}
    for i, actor in enumerate(unique_actors):
        nid2e_dict[i + acc] = ("actor", actor)
    acc += num_actors
    type_accs["writer"] = acc
    writer2nid = {writer: i + acc for i, writer in enumerate(unique_writers)}
    for i, writer in enumerate(unique_writers):
        nid2e_dict[i + acc] = ("writer", writer)
    acc += num_writers
    type_accs["tid"] = acc
    tag2nid = {tid: i + acc for i, tid in enumerate(unique_tids)}
    for i, tag in enumerate(unique_tids):
        nid2e_dict[i + acc] = ("tid", tag)
    acc += num_tids
    type_accs["genome_tid"] = acc
    genome_tag2nid = {
        genome_tid: i + acc for i, genome_tid in enumerate(unique_genome_tids)
    }
    for i, genome_tag in enumerate(unique_genome_tids):
        nid2e_dict[i + acc] = ("genome_tid", genome_tag)
    e2nid_dict = {
        "uid": uid2nid,
        "iid": iid2nid,
        "genre": genre2nid,
        "year": year2nid,
        "director": director2nid,
        "actor": actor2nid,
        "writer": writer2nid,
        "tid": tag2nid,
        "genome_tid": genome_tag2nid,
    }
    dataset_property_dict["e2nid_dict"] = e2nid_dict
    dataset_property_dict["nid2e_dict"] = nid2e_dict

    #########################  create graphs  #########################
    edge_index_nps = {}
    print("Creating item attribute edges...")
    inids = [e2nid_dict["iid"][iid] for iid in movies.iid]
    year_nids = [e2nid_dict["year"][year] for year in movies.year]
    year2item_edge_index_np = np.vstack((np.array(year_nids), np.array(inids)))

    genre_nids = []
    inids = []
    for genre in unique_genres:
        iids = movies[movies[genre]].iid
        inids += [e2nid_dict["iid"][iid] for iid in iids]
        genre_nids += [e2nid_dict["genre"][genre] for _ in range(iids.shape[0])]
    genre2item_edge_index_np = np.vstack((np.array(genre_nids), np.array(inids)))

    inids = [e2nid_dict["iid"][iid] for iid in movies.iid]
    directors_list = [
        [director for director in directors.split(",") if director != ""]
        for directors in movies.directors
    ]
    directors_nids = [
        [e2nid_dict["director"][director] for director in directors]
        for directors in directors_list
    ]
    directors_nids = list(itertools.chain.from_iterable(directors_nids))
    d_inids = [
        [i_nid for _ in range(len(directors_list[idx]))]
        for idx, i_nid in enumerate(inids)
    ]
    d_inids = list(itertools.chain.from_iterable(d_inids))
    director2item_edge_index_np = np.vstack(
        (np.array(directors_nids), np.array(d_inids))
    )

    actors_list = [
        [actor for actor in actors.split(",") if actor != ""]
        for actors in movies.actors
    ]
    actor_nids = [
        [e2nid_dict["actor"][actor] for actor in actors] for actors in actors_list
    ]
    actor_nids = list(itertools.chain.from_iterable(actor_nids))
    a_inids = [
        [i_nid for _ in range(len(actors_list[idx]))] for idx, i_nid in enumerate(inids)
    ]
    a_inids = list(itertools.chain.from_iterable(a_inids))
    actor2item_edge_index_np = np.vstack((np.array(actor_nids), np.array(a_inids)))

    writers_list = [
        [writer for writer in writers.split(",") if writer != ""]
        for writers in movies.writers
    ]
    writer_nids = [
        [e2nid_dict["writer"][writer] for writer in writers] for writers in writers_list
    ]
    writer_nids = list(itertools.chain.from_iterable(writer_nids))
    w_inids = [
        [i_nid for _ in range(len(writers_list[idx]))]
        for idx, i_nid in enumerate(inids)
    ]
    w_inids = list(itertools.chain.from_iterable(w_inids))
    writer2item_edge_index_np = np.vstack((np.array(writer_nids), np.array(w_inids)))
    edge_index_nps["year2item"] = year2item_edge_index_np
    edge_index_nps["genre2item"] = genre2item_edge_index_np
    edge_index_nps["director2item"] = director2item_edge_index_np
    edge_index_nps["actor2item"] = actor2item_edge_index_np
    edge_index_nps["writer2item"] = writer2item_edge_index_np

    inids = [e2nid_dict["iid"][iid] for iid in genome_tagging.iid]
    genome_tnids = [
        e2nid_dict["genome_tid"][genome_tid] for genome_tid in genome_tagging.genome_tid
    ]
    genome_tag2item_edge_index_np = np.vstack((np.array(genome_tnids), np.array(inids)))
    edge_index_nps["genome_tag2item"] = genome_tag2item_edge_index_np

    unids = [e2nid_dict["uid"][uid] for uid in tagging.uid]
    tnids = [e2nid_dict["tid"][tid] for tid in tagging.tid]
    inids = [e2nid_dict["iid"][iid] for iid in tagging.iid]
    tag2user_edge_index_np = np.vstack((np.array(tnids), np.array(unids)))
    tag2item_edge_index_np = np.vstack((np.array(tnids), np.array(inids)))
    edge_index_nps["tag2user"] = tag2user_edge_index_np
    edge_index_nps["tag2item"] = tag2item_edge_index_np

    print("Creating rating property edges...")
    test_pos_unid_inid_map, neg_unid_inid_map = {}, {}

    rating_np = np.zeros((0,))
    user2item_edge_index_np = np.zeros((2, 0))
    sorted_ratings = ratings.sort_values("uid")
    pbar = tqdm.tqdm(unique_uids, total=len(unique_uids))
    for uid in pbar:
        pbar.set_description("Creating the edges for the user {}".format(uid))
        uid_ratings = sorted_ratings[sorted_ratings.uid == uid].sort_values("timestamp")
        uid_iids = uid_ratings.iid.to_numpy()
        uid_ratings = uid_ratings.rating.to_numpy()

        unid = e2nid_dict["uid"][uid]
        train_pos_uid_iids = list(uid_iids[:-1])  # Use leave one out setup
        train_pos_uid_ratings = uid_ratings[:-1]
        train_pos_uid_inids = [e2nid_dict["iid"][iid] for iid in train_pos_uid_iids]
        test_pos_uid_iids = list(uid_iids[-1:])
        test_pos_uid_inids = [e2nid_dict["iid"][iid] for iid in test_pos_uid_iids]
        neg_uid_iids = list(set(unique_iids) - set(uid_iids))
        neg_uid_inids = [e2nid_dict["iid"][iid] for iid in neg_uid_iids]

        test_pos_unid_inid_map[unid] = test_pos_uid_inids
        neg_unid_inid_map[unid] = neg_uid_inids

        unid_user2item_edge_index_np = np.array(
            [[unid for _ in range(len(train_pos_uid_inids))], train_pos_uid_inids]
        )
        user2item_edge_index_np = np.hstack(
            [user2item_edge_index_np, unid_user2item_edge_index_np]
        )

        rating_np = np.concatenate([rating_np, train_pos_uid_ratings])
    dataset_property_dict["rating_np"] = rating_np
    edge_index_nps["user2item"] = user2item_edge_index_np

    dataset_property_dict["edge_index_nps"] = edge_index_nps
    (
        dataset_property_dict["test_pos_unid_inid_map"],
        dataset_property_dict["neg_unid_inid_map"],
    ) = (test_pos_unid_inid_map, neg_unid_inid_map)

    print("Building edge type map...")
    edge_type_dict = {
        edge_type: edge_type_idx
        for edge_type_idx, edge_type in enumerate(list(edge_index_nps.keys()))
    }
    dataset_property_dict["edge_type_dict"] = edge_type_dict
    dataset_property_dict["num_edge_types"] = len(list(edge_index_nps.keys()))

    print("Building the item occurrence map...")
    item_count = ratings["iid"].value_counts()
    item_nid_occs = {}
    for iid in unique_iids:
        item_nid_occs[e2nid_dict["iid"][iid]] = item_count[iid]
    dataset_property_dict["item_nid_occs"] = item_nid_occs

    # New functionality for pytorch geometric like dataset
    dataset_property_dict["types"] = types
    dataset_property_dict["num_nodes_dict"] = num_nodes_dict
    dataset_property_dict["type_accs"] = type_accs

    return dataset_property_dict


class MovieLens(Dataset):
    edge_hist = {}
    edge_last_use = {}
    testing_edges = {}
    url = "http://files.grouplens.org/datasets/movielens/"

    def __init__(
        self, root, name, transform=None, pre_transform=None, pre_filter=None, **kwargs
    ):
        self.token = kwargs["token"]
        self.name = name.lower()
        self.type = kwargs["type"]
        assert self.name in ["25m", "latest-small"]
        assert self.type in ["hete"]
        self.num_core = kwargs["num_core"]
        self.num_feat_core = kwargs["num_feat_core"]

        self.entity_aware = kwargs["entity_aware"]

        self.num_negative_samples = kwargs["num_negative_samples"]
        self.sampling_strategy = kwargs["sampling_strategy"]
        self.cf_loss_type = kwargs["cf_loss_type"]

        self.timeframe = kwargs["timeframe"]
        self.num_timeframes = kwargs["num_timeframes"]
        self.equal_timespan_timeframes = kwargs["equal_timespan_timeframes"]
        self.start_timeframe = kwargs["start_timeframe"]

        self.continual_aspect = kwargs["continual_aspect"]
        self.future_testing = kwargs["future_testing"]
        self.last_task_accuracy = kwargs["last_task_accuracy"]

        super(MovieLens, self).__init__(root, transform, pre_transform, pre_filter)

        with open(self.processed_paths[0], "rb") as f:  # Read the class property
            dataset_property_dict = pickle.load(f)
        for k, v in dataset_property_dict.items():
            self[k] = v

        print("Dataset loaded!")

    @property
    def raw_file_names(self):
        return "ml-{}.zip".format(self.name.lower())

    @property
    def processed_file_names(self):
        return ["ml_{}_{}_{}.pkl".format(self.name, self.build_suffix(), self.token)]

    def download(self):
        path = download_url(self.url + self.raw_file_names, self.raw_dir)
        extract_zip(path, self.raw_dir)

    def process(self):
        if self.name == "25m":
            try:
                movies = pd.read_csv(
                    join(self.processed_dir, "movies.csv"), sep=";"
                ).fillna("")
                ratings = pd.read_csv(join(self.processed_dir, "ratings.csv"), sep=";")
                tagging = pd.read_csv(join(self.processed_dir, "tagging.csv"), sep=";")
                genome_tagging = pd.read_csv(
                    join(self.processed_dir, "genome_tagging.csv"), sep=";"
                )
                print("Read data frame from {}!".format(self.processed_dir))
            except:
                unzip_raw_dir = join(self.raw_dir, "ml-{}".format(self.name))
                print(
                    "Data frame not found in {}! Read from raw data and preprocessing from {}!".format(
                        self.processed_dir, unzip_raw_dir
                    )
                )

                raw_movies_path = join(self.raw_dir, "raw_movies.csv")
                raw_ratings_path = join(self.raw_dir, "raw_ratings.csv")
                raw_tagging_path = join(self.raw_dir, "raw_tagging.csv")
                raw_genome_scores_path = join(self.raw_dir, "raw_genome_scores.csv")
                raw_genome_tags_path = join(self.raw_dir, "raw_genome_tags.csv")
                if not (
                    isfile(raw_movies_path)
                    and isfile(raw_ratings_path)
                    and isfile(raw_tagging_path)
                    and isfile(raw_genome_scores_path)
                    and isfile(raw_genome_tags_path)
                ):
                    print(
                        "Raw files not found! Reading directories and actors from api!"
                    )
                    movies, ratings, tagging, genome_scores, genome_tags = parse_ml25m(
                        unzip_raw_dir
                    )
                    save_df(movies, raw_movies_path)
                    save_df(ratings, raw_ratings_path)
                    save_df(tagging, raw_tagging_path)
                    save_df(genome_scores, raw_genome_scores_path)
                    save_df(genome_tags, raw_genome_tags_path)
                else:
                    print("Raw files loaded!")
                    movies = pd.read_csv(raw_movies_path, sep=";").fillna("")
                    ratings = pd.read_csv(raw_ratings_path, sep=";")
                    tagging = pd.read_csv(raw_tagging_path, sep=";")
                    genome_scores = pd.read_csv(raw_genome_scores_path, sep=";")
                    genome_tags = pd.read_csv(raw_genome_tags_path, sep=";")

                # Remove duplicates
                movies = movies.drop_duplicates()
                ratings = ratings.drop_duplicates()
                tagging = tagging.drop_duplicates()
                genome_scores = genome_scores.drop_duplicates()
                genome_tags = genome_tags.drop_duplicates()

                ratings = ratings[ratings.timestamp > 1514764799]  # 2M interactions

                # Sync
                movies = movies[movies.iid.isin(ratings.iid.unique())]
                ratings = ratings[ratings.iid.isin(movies.iid.unique())]
                tagging = tagging[tagging.iid.isin(ratings.iid.unique())]
                tagging = tagging[tagging.uid.isin(ratings.uid.unique())]
                genome_scores = genome_scores[
                    genome_scores.iid.isin(ratings.iid.unique())
                ]
                genome_scores = genome_scores[
                    genome_scores.genome_tid.isin(genome_tags.genome_tid.unique())
                ]
                genome_tags = genome_tags[
                    genome_tags.genome_tid.isin(genome_scores.genome_tid.unique())
                ]

                # Remove infrequent movies
                movie_count = ratings["iid"].value_counts()
                movie_count.name = "movie_count"
                ratings = ratings[
                    ratings.join(movie_count, on="iid").movie_count > self.num_core
                ]

                # Remove infrequent users
                user_count = ratings["uid"].value_counts()
                user_count.name = "user_count"
                ratings = ratings.join(user_count, on="uid")
                ratings = ratings[ratings.user_count > self.num_core]
                ratings = ratings[ratings.user_count < 30 * self.num_core]
                ratings = ratings.drop(columns=["user_count"])

                # Sync
                movies = movies[movies.iid.isin(ratings.iid.unique())]
                tagging = tagging[tagging.iid.isin(ratings.iid.unique())]
                tagging = tagging[tagging.uid.isin(ratings.uid.unique())]
                genome_scores = genome_scores[
                    genome_scores.iid.isin(ratings.iid.unique())
                ]
                genome_tags = genome_tags[
                    genome_tags.genome_tid.isin(genome_scores.genome_tid.unique())
                ]

                # Remove infrequent tags
                tag_count = tagging["tag"].value_counts()
                tag_count.name = "tag_count"
                tagging = tagging[
                    tagging.join(tag_count, on="tag").tag_count > self.num_feat_core
                ]

                # Remove infrequent genome tags
                genome_tagging = genome_scores[genome_scores.relevance > 0.5]
                genome_tag_count = genome_tagging["genome_tid"].value_counts()
                genome_tag_count.name = "genome_tag_count"
                genome_tagging = genome_tagging[
                    genome_tagging.join(genome_tag_count, "genome_tid").genome_tag_count
                    > self.num_feat_core
                ]

                # Reindex the uid and iid in case of missing values
                (
                    movies,
                    ratings,
                    tagging,
                    tags,
                    genome_tagging,
                    genome_tags,
                ) = reindex_df_ml25m(
                    movies, ratings, tagging, genome_tagging, genome_tags
                )

                # Drop the infrequent writer, actor and directors
                movies = drop_infrequent_concept_from_str(
                    movies, "writers", self.num_feat_core
                )
                movies = drop_infrequent_concept_from_str(
                    movies, "directors", self.num_feat_core
                )
                movies = drop_infrequent_concept_from_str(
                    movies, "actors", self.num_feat_core
                )

                # filter the years
                years = movies.year.to_numpy()
                years[years < 1950] = 1950
                movies["year"] = years
                if self.type == "hete":
                    years = movies.year.to_numpy().astype(np.int)
                    min_year = min(years)
                    max_year = max(years)
                    num_years = (max_year - min_year) // 10
                    discretized_years = [
                        min_year + i * 10 for i in range(num_years + 1)
                    ]
                    for i in range(len(discretized_years) - 1):
                        years[
                            (discretized_years[i] <= years)
                            & (years < discretized_years[i + 1])
                        ] = str(discretized_years[i])
                    years[years < discretized_years[0]] = discretized_years[0]
                    years[years >= discretized_years[-1]] = discretized_years[-1]

                    movies["year"] = years

                # save dfs
                print("Saving processed csv...")
                save_df(tags, join(self.processed_dir, "tags.csv"))
                save_df(tagging, join(self.processed_dir, "tagging.csv"))
                save_df(genome_tagging, join(self.processed_dir, "genome_tagging.csv"))
                save_df(genome_tags, join(self.processed_dir, "genome_tags.csv"))
                save_df(movies, join(self.processed_dir, "movies.csv"))
                save_df(ratings, join(self.processed_dir, "ratings.csv"))

            # Generate and save graph
            if self.type == "hete":
                dataset_property_dict = generate_ml25m_hete_graph(
                    movies, ratings, tagging, genome_tagging
                )
            else:
                raise NotImplementedError
            with open(self.processed_paths[0], "wb") as f:
                pickle.dump(dataset_property_dict, f)
        elif self.name == "latest-small":
            try:
                movies = pd.read_csv(
                    join(self.processed_dir, "movies.csv"), sep=";"
                ).fillna("")
                ratings = pd.read_csv(join(self.processed_dir, "ratings.csv"), sep=";")
                tagging = pd.read_csv(join(self.processed_dir, "tagging.csv"), sep=";")
                print("Read data frame from {}!".format(self.processed_dir))
            except:
                unzip_raw_dir = join(self.raw_dir, "ml-{}".format(self.name))
                print(
                    "Data frame not found in {}! Read from raw data and preprocessing from {}!".format(
                        self.processed_dir, unzip_raw_dir
                    )
                )

                raw_movies_path = join(self.raw_dir, "raw_movies.csv")
                raw_ratings_path = join(self.raw_dir, "raw_ratings.csv")
                raw_tagging_path = join(self.raw_dir, "raw_tagging.csv")
                if not (
                    isfile(raw_movies_path)
                    and isfile(raw_ratings_path)
                    and isfile(raw_tagging_path)
                ):
                    print(
                        "Raw files not found! Reading directories and actors from api!"
                    )
                    movies, ratings, tagging = parse_mlsmall(unzip_raw_dir)
                    save_df(movies, raw_movies_path)
                    save_df(ratings, raw_ratings_path)
                    save_df(tagging, raw_tagging_path)
                else:
                    print("Raw files loaded!")
                    movies = pd.read_csv(raw_movies_path, sep=";").fillna("")
                    ratings = pd.read_csv(raw_ratings_path, sep=";")
                    tagging = pd.read_csv(raw_tagging_path, sep=";")

                # Remove duplicates
                movies = movies.drop_duplicates()
                ratings = ratings.drop_duplicates()
                tagging = tagging.drop_duplicates()

                # Sync
                movies = movies[movies.iid.isin(ratings.iid.unique())]
                ratings = ratings[ratings.iid.isin(movies.iid.unique())]
                tagging = tagging[tagging.iid.isin(ratings.iid.unique())]
                tagging = tagging[tagging.uid.isin(ratings.uid.unique())]

                # ratings = ratings[:len(ratings)//10]

                # Remove infrequent movies
                movie_count = ratings["iid"].value_counts()
                movie_count.name = "movie_count"
                ratings = ratings[
                    ratings.join(movie_count, on="iid").movie_count > self.num_core
                ]

                # Remove infrequent users
                user_count = ratings["uid"].value_counts()
                user_count.name = "user_count"
                ratings = ratings[
                    ratings.join(user_count, on="uid").user_count > self.num_core
                ]

                # Sync
                movies = movies[movies.iid.isin(ratings.iid.unique())]
                tagging = tagging[tagging.iid.isin(ratings.iid.unique())]
                tagging = tagging[tagging.uid.isin(ratings.uid.unique())]

                # Remove infrequent tags
                tag_count = tagging["tag"].value_counts()
                tag_count.name = "tag_count"
                tagging = tagging[
                    tagging.join(tag_count, on="tag").tag_count > self.num_feat_core
                ]

                # filter the years
                years = movies.year.to_numpy()
                years[years < 1950] = 1950
                movies["year"] = years
                if self.type == "hete":
                    years = movies.year.to_numpy().astype(np.int)
                    min_year = min(years)
                    max_year = max(years)
                    num_years = (max_year - min_year) // 10
                    discretized_years = [
                        min_year + i * 10 for i in range(num_years + 1)
                    ]
                    for i in range(len(discretized_years) - 1):
                        years[
                            (discretized_years[i] <= years)
                            & (years < discretized_years[i + 1])
                        ] = str(discretized_years[i])
                    years[years < discretized_years[0]] = discretized_years[0]
                    years[years >= discretized_years[-1]] = discretized_years[-1]

                    movies["year"] = years

                # Reindex the uid and iid in case of missing values
                movies, ratings, tagging, tags = reindex_df_mlsmall(
                    movies, ratings, tagging
                )

                # Drop the infrequent writer, actor and directors
                movies = drop_infrequent_concept_from_str(
                    movies, "writers", self.num_feat_core
                )
                movies = drop_infrequent_concept_from_str(
                    movies, "directors", self.num_feat_core
                )
                movies = drop_infrequent_concept_from_str(
                    movies, "actors", self.num_feat_core
                )

                # calculate start & stop s.t. timeframe timestamps in [start, stop)
                if self.timeframe >= 0:
                    if self.equal_timespan_timeframes:
                        min_timestamp = ratings.timestamp.min()
                        max_timestamp = ratings.timestamp.max() + 1

                        diff = (max_timestamp - min_timestamp) / self.num_timeframes
                        self.start = min_timestamp + self.timeframe * diff
                        self.stop = self.start + diff
                        self.init_time = min_timestamp + self.start_timeframe * diff

                    else:
                        ratings = ratings.sort_values("timestamp")
                        # ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
                        # ratings.timestamp = ratings.index
                        timeframe_size = round(len(ratings) / self.num_timeframes)
                        start_id = self.timeframe * timeframe_size
                        stop_id = (
                            min((self.timeframe + 1) * timeframe_size, len(ratings)) - 1
                        )
                        init_id = self.start_timeframe * timeframe_size
                        print(f"selecting {start_id} - {stop_id}")
                        self.start = ratings.iloc[start_id]["timestamp"]
                        self.stop = ratings.iloc[stop_id]["timestamp"] + 0.5
                        self.init_time = ratings.iloc[init_id]["timestamp"]

                # save dfs
                print("Saving processed csv...")
                save_df(tags, join(self.processed_dir, "tags.csv"))
                save_df(tagging, join(self.processed_dir, "tagging.csv"))
                save_df(movies, join(self.processed_dir, "movies.csv"))
                save_df(ratings, join(self.processed_dir, "ratings.csv"))

            # Generate and save graph
            if self.type == "hete":
                dataset_property_dict = generate_mlsmall_hete_graph(
                    movies,
                    ratings,
                    tagging,
                    self.start,
                    self.stop,
                    self.init_time,
                    self.continual_aspect in ["online", "single"],
                    self.future_testing,
                    self.last_task_accuracy,
                    self.testing_edges,
                )

                if self.continual_aspect != "retrained":
                    ratings = ratings[ratings.timestamp >= self.start]

                ratings = ratings[ratings.timestamp < self.stop]

                # remember the provenence of the ratings
                for rating in ratings.itertuples():
                    self.edge_hist[
                        (rating.uid, rating.iid + dataset_property_dict["num_uids"])
                    ] = self.timeframe

                if len(ratings) == 0 or (
                    self.continual_aspect == "pretrained" and self.timeframe > 0
                ):
                    print("SKIPPED TIMEFRAME")
                    self.skip_timeframe = True
                else:
                    self.skip_timeframe = False
            else:
                raise NotImplementedError
            with open(self.processed_paths[0], "wb") as f:
                pickle.dump(dataset_property_dict, f)

    def build_suffix(self):
        return "core_{}_type_{}".format(self.num_core, self.type)

    def kg_negative_sampling(self):
        """
        Replace tail entities in existing triples with random entities
        """
        print("KG negative sampling...")
        pos_edge_index_r_nps = [
            (
                edge_index,
                np.ones((edge_index.shape[1], 1)) * self.edge_type_dict[edge_type],
            )
            for edge_type, edge_index in self.edge_index_nps.items()
        ]
        pos_edge_index_trans_np = np.hstack([_[0] for _ in pos_edge_index_r_nps]).T
        pos_r_np = np.vstack([_[1] for _ in pos_edge_index_r_nps])
        neg_t_np = np.random.randint(
            low=0, high=self.num_nodes, size=(pos_edge_index_trans_np.shape[0], 1)
        )
        train_data_np = np.hstack([pos_edge_index_trans_np, neg_t_np, pos_r_np])
        train_data_t = torch.from_numpy(train_data_np).long()
        shuffle_idx = torch.randperm(train_data_t.shape[0])
        self.train_data = train_data_t[shuffle_idx]
        self.train_data_length = train_data_t.shape[0]

    def cf_negative_sampling(self, last_emb, crt_emb, theta, epoch):
        """
        Replace positive items with random/unseen items
        """
        print("CF negative sampling...")
        pos_edge_index_trans_np = self.edge_index_nps["user2item"].T
        print("before:")
        print(len(pos_edge_index_trans_np))

        # edge is from current timeframe
        def is_crt(e):
            n0 = int(e[0].item())
            n1 = int(e[1].item())
            return self.edge_hist.get((n0, n1), -1) == self.timeframe

        # importance heuristic
        def h(e):
            # return 1
            # n0 = int(e[0].item())
            # n1 = int(e[1].item())
            # last_e = torch.cat([last_emb[n0], last_emb[n1]], dim=-1)
            # crt_e  = torch.cat([crt_emb[n0], crt_emb[n1]], dim=-1)
            # d = torch.norm(crt_e - last_e)
            d = np.random.random()

            return d

        # age of edge
        def age(e):
            n0 = int(e[0].item())
            n1 = int(e[1].item())
            return self.timeframe - self.edge_last_use.get((n0, n1), 0) - 1

        def edge_emb(e):
            n0 = int(e[0].item())
            n1 = int(e[1].item())
            return torch.cat([crt_emb[n0], crt_emb[n1]], dim=-1)

        if last_emb is not None and self.continual_aspect == "continual":
            if epoch == 1:
                # ro = 0
                # save importances
                hs = {e.tobytes(): (is_crt(e), h(e)) for e in pos_edge_index_trans_np}
                # hs = {e.tobytes() : h(e) for e in pos_edge_index_trans_np}

                # sort by importance
                pos_edge_index_trans_np = np.array(
                    sorted(
                        pos_edge_index_trans_np,
                        key=lambda e: hs[e.tobytes()],
                        reverse=True,
                    )
                )

                len_ratings = sum(
                    self.edge_hist[(int(e0.item()), int(e1.item()))] == self.timeframe
                    for e0, e1 in pos_edge_index_trans_np
                )

                no_samples = min(
                    len(pos_edge_index_trans_np), round(theta * len_ratings)
                )
                print(f"no_samples: {no_samples}")

                # below there are the different heuristics I tried out
                # edge_embs = torch.stack([edge_emb(e) for e in pos_edge_index_trans_np])
                # distances = torch.cdist(edge_embs, edge_embs)

                # selected_indeces = []
                # distances_to_se = torch.zeros(len(edge_embs)).to('cuda')

                # for _ in range(no_samples):
                #     index = np.random.randint(no_samples)
                #     if len(selected_indeces) > 0:
                #         distances_to_se += distances[:, selected_indeces[-1]]
                #         index = torch.argmax(distances_to_se).item()

                #     selected_indeces.append(index)

                # pos_edge_index_trans_np = pos_edge_index_trans_np[selected_indeces]
                pos_edge_index_trans_np = pos_edge_index_trans_np[:no_samples]
                # no_samples -= self.len_ratings
                # ch_no_samples = int(ro * no_samples)
                # rr_no_samples = no_samples - ch_no_samples

                # imp_samp = pos_edge_index_trans_np[:self.len_ratings + int(ro * no_samples)]

                # rand_samp = pos_edge_index_trans_np[self.len_ratings + int(ro * no_samples):]
                # inds = np.random.choice(
                #     no_samples,
                #     no_samples - int(ro * no_samples)
                # )
                # rand_samp = rand_samp[inds]

                # pos_edge_index_trans_np_old = np.array([
                #     e for e in pos_edge_index_trans_np if not is_crt(e)
                # ])

                # pos_edge_index_trans_np_new = np.array([
                #     e for e in pos_edge_index_trans_np if is_crt(e)
                # ])

                # print([
                #     self.edge_hist.get((e[0], e[1]), -1)
                #     for e in pos_edge_index_trans_np_old])
                # print([
                #     self.edge_last_use.get((e[0], e[1]), -2)
                #     for e in pos_edge_index_trans_np_old])

                # p = torch.tensor([
                #     1 if self.edge_hist.get((e[0], e[1]), -1) == self.edge_last_use.get((e[0], e[1]), 0) else 0
                #     for e in pos_edge_index_trans_np_old], dtype=torch.double)
                # p = torch.tensor([
                #     self.timeframe - self.edge_hist.get((e[0], e[1]), 0) <= 2
                #     for e in pos_edge_index_trans_np_old], dtype=torch.double)
                # print(p)
                # p /= sum(p)
                # print(p)
                # inds = np.random.choice(
                #     len(pos_edge_index_trans_np_old),
                #     no_samples,
                #     p = p,
                #     replace=True
                # )
                # pos_edge_index_trans_np_old = pos_edge_index_trans_np_old[inds]

                # hs = torch.tensor([h(e) for e in pos_edge_index_trans_np_old], dtype=torch.double)
                # hs /= sum(hs)
                # inds = np.random.choice(
                #     len(pos_edge_index_trans_np_old),
                #     ch_no_samples,
                #     p = hs,
                #     replace=True
                # )
                # ch_samples = pos_edge_index_trans_np_old[inds]

                # pos_edge_index_trans_np_old = np.delete(
                #     pos_edge_index_trans_np_old,
                #     inds,
                #     0
                # )

                # ages = torch.tensor([2**age(e) for e in pos_edge_index_trans_np_old], dtype=torch.double)
                # ages /= sum(ages)
                # inds = np.random.choice(
                #     len(pos_edge_index_trans_np_old),
                #     no_samples,
                #     p = ages,
                #     replace=True
                # )
                # pos_edge_index_trans_np_old = pos_edge_index_trans_np_old[inds]

                # imp = torch.tensor([h(e) for e in pos_edge_index_trans_np_old], dtype=torch.double)
                # imp /= sum(imp)
                # inds = np.random.choice(
                #     len(pos_edge_index_trans_np_old),
                #     no_samples,
                #     p = imp,
                #     replace=True
                # )
                # pos_edge_index_trans_np_old = pos_edge_index_trans_np_old[inds]

                # ages = torch.tensor([2**age(e) for e in pos_edge_index_trans_np_old], dtype=torch.double)
                # ages /= sum(ages)
                # inds = np.random.choice(
                #     len(pos_edge_index_trans_np_old),
                #     rr_no_samples,
                #     p = ages,
                #     replace=True
                # )
                # rr_samples = pos_edge_index_trans_np_old[inds]

                # eps = 0
                # T = 2**16
                # b = 1

                # imps = torch.tensor([h(e) * (b ** age(e)) for e in pos_edge_index_trans_np_old], dtype=torch.double)
                # imps =  imps * T + eps
                # print(f'imps: {imps}, min: {min(imps)}, max: {max(imps)}, mean: {torch.mean(imps)}')
                # p = torch.softmax(imps, dim=0)
                # p /= sum(p)
                # print(f'probs: {p}, min: {min(p)}, max: {max(p)}')

                # inds = np.random.choice(
                #     len(pos_edge_index_trans_np_old),
                #     no_samples,
                #     p = p,
                #     replace = True
                # )
                # pos_edge_index_trans_np_old = pos_edge_index_trans_np_old[inds]

                # pos_edge_index_trans_np = np.concatenate(
                #     (pos_edge_index_trans_np_new, pos_edge_index_trans_np_old)
                # )

                # pos_edge_index_trans_np = np.concatenate(
                #     (pos_edge_index_trans_np_new, ch_samples, rr_samples)
                # )

                self.pos_edge_index_trans_np = pos_edge_index_trans_np

                # calculate edge selection distribution and save the last use of the edges
                edge_dist = {}
                for edge in pos_edge_index_trans_np:
                    e0 = int(edge[0].item())
                    e1 = int(edge[1].item())
                    timeframe_no = self.edge_hist.get((e0, e1), "future")
                    edge_dist[timeframe_no] = edge_dist.get(timeframe_no, 0) + 1
                    self.edge_last_use[(e0, e1)] = self.timeframe

                print(
                    "edge dist: "
                    + "".join(
                        [
                            f"{t}:{edge_dist.get(t, 0)} "
                            for t in range(self.num_timeframes)
                        ]
                    )
                )
            else:
                pos_edge_index_trans_np = self.pos_edge_index_trans_np

        print("after:")
        print(len(pos_edge_index_trans_np))

        num_interactions = pos_edge_index_trans_np.shape[0]
        if self.cf_loss_type == "BCE":
            pos_samples_np = np.hstack(
                [
                    pos_edge_index_trans_np,
                    np.ones((pos_edge_index_trans_np.shape[0], 1)),
                ]
            )
            if self.sampling_strategy == "random":
                neg_samples_np = np.hstack(
                    [
                        np.repeat(
                            pos_samples_np[:, 0].reshape(-1, 1),
                            repeats=self.num_negative_samples,
                            axis=0,
                        ),
                        np.random.randint(
                            low=self.type_accs["iid"],
                            high=self.type_accs["iid"] + self.num_iids,
                            size=(num_interactions * self.num_negative_samples, 1),
                        ),
                        torch.zeros((num_interactions * self.num_negative_samples, 1)),
                    ]
                )
            elif self.sampling_strategy == "unseen":
                neg_inids = []
                u_nids = pos_samples_np[:, 0]
                p_bar = tqdm.tqdm(u_nids)
                for u_nid in p_bar:
                    negative_inids = (
                        self.test_pos_unid_inid_map.get(u_nid, [])
                        + self.neg_unid_inid_map[u_nid]
                    )
                    negative_inids = np.random.choice(
                        negative_inids, size=(self.num_negative_samples, 1)
                    )
                    neg_inids.append(negative_inids)
                neg_samples_np = np.hstack(
                    [
                        np.repeat(
                            pos_samples_np[:, 0].reshape(-1, 1),
                            repeats=self.num_negative_samples,
                            axis=0,
                        ),
                        np.vstack(neg_inids),
                        np.zeros((num_interactions * self.num_negative_samples, 1)),
                    ]
                )
            else:
                raise NotImplementedError
            train_data_np = np.vstack([pos_samples_np, neg_samples_np])
        elif self.cf_loss_type == "MSE":
            train_data_np = np.hstack(
                [pos_edge_index_trans_np, self.rating_np.reshape(-1, 1)]
            )
        elif self.cf_loss_type == "BPR":
            train_data_np = np.repeat(
                pos_edge_index_trans_np, repeats=self.num_negative_samples, axis=0
            )
            if self.sampling_strategy == "random":
                neg_inid_np = np.random.randint(
                    low=self.type_accs["iid"],
                    high=self.type_accs["iid"] + self.num_iids,
                    size=(num_interactions * self.num_negative_samples, 1),
                )
            elif self.sampling_strategy == "unseen":
                neg_inids = []
                u_nids = pos_edge_index_trans_np[:, 0]

                p_bar = tqdm.tqdm(u_nids)
                # TODO: repeat negative samples based on theta
                for u_nid in p_bar:
                    negative_inids = (
                        self.test_pos_unid_inid_map.get(u_nid, [])
                        + self.neg_unid_inid_map[u_nid]
                    )
                    negative_inids = rd.choices(
                        negative_inids, k=self.num_negative_samples
                    )
                    negative_inids = np.array(negative_inids, dtype=np.long).reshape(
                        -1, 1
                    )
                    # repeat negative samples theta times
                    # print('Dimension of negative_inids before: ', negative_inids.shape)
                    negative_inids = np.repeat(negative_inids, repeats=theta, axis=0)
                    # print('Dimension of negative_inids after: ', negative_inids.shape)
                    neg_inids.append(negative_inids)
                neg_inid_np = np.vstack(neg_inids)
            else:
                raise NotImplementedError
            train_data_np = np.repeat(train_data_np, repeats=theta, axis=0)
            train_data_np = np.hstack([train_data_np, neg_inid_np])

            if self.entity_aware and not hasattr(self, "iid_feat_nids"):
                # add entity aware data to batches
                movies = pd.read_csv(
                    join(self.processed_dir, "movies.csv"), sep=";"
                ).fillna("")
                tagging = pd.read_csv(join(self.processed_dir, "tagging.csv"), sep=";")
                if self.name == "25m":
                    genome_tagging = pd.read_csv(
                        join(self.processed_dir, "genome_tagging.csv"), sep=";"
                    )

                # Build item entity
                iid_feat_nids = []
                pbar = tqdm.tqdm(self.unique_iids, total=len(self.unique_iids))
                for iid in pbar:
                    pbar.set_description("Sampling item entities...")

                    feat_nids = []

                    year_nid = self.e2nid_dict["year"][
                        movies[movies.iid == iid]["year"].item()
                    ]
                    feat_nids.append(year_nid)

                    genre_nids = [
                        self.e2nid_dict["genre"][genre]
                        for genre in self.unique_genres
                        if movies[movies.iid == iid][genre].item()
                    ]
                    feat_nids += genre_nids

                    actor_nids = [
                        self.e2nid_dict["actor"][actor]
                        for actor in movies[movies.iid == iid]["actors"]
                        .item()
                        .split(",")
                        if actor != ""
                    ]
                    feat_nids += actor_nids

                    director_nids = [
                        self.e2nid_dict["director"][director]
                        for director in movies[movies.iid == iid]["directors"]
                        .item()
                        .split(",")
                        if director != ""
                    ]
                    feat_nids += director_nids

                    writer_nids = [
                        self.e2nid_dict["writer"][writer]
                        for writer in movies[movies.iid == iid]["writers"]
                        .item()
                        .split(",")
                        if writer != ""
                    ]
                    feat_nids += writer_nids

                    tag_nids = [
                        self.e2nid_dict["tid"][tid]
                        for tid in tagging[tagging.iid == iid].tid
                    ]
                    feat_nids += tag_nids
                    if self.name == "25m":
                        genome_tag_nids = [
                            self.e2nid_dict["genome_tid"][genome_tid]
                            for genome_tid in genome_tagging[
                                genome_tagging.iid == iid
                            ].genome_tid
                        ]
                        feat_nids += genome_tag_nids
                    iid_feat_nids.append(feat_nids)
                self.iid_feat_nids = iid_feat_nids

                # Build user entity
                uid_feat_nids = []
                pbar = tqdm.tqdm(self.unique_uids, total=len(self.unique_uids))
                for uid in pbar:
                    pbar.set_description("Sampling user entities...")
                    feat_nids = []

                    tag_nids = [
                        self.e2nid_dict["tid"][tid]
                        for tid in tagging[tagging.uid == uid].tid
                    ]
                    feat_nids += tag_nids
                    uid_feat_nids.append(feat_nids)
                self.uid_feat_nids = uid_feat_nids
        else:
            raise NotImplementedError
        train_data_t = torch.from_numpy(train_data_np).long()
        shuffle_idx = torch.randperm(train_data_t.shape[0])
        self.train_data = train_data_t[shuffle_idx]
        self.train_data_length = train_data_t.shape[0]

        user_item_pairs = list(zip(self.train_data.t()[0], self.train_data.t()[1]))

        # For each tuple in the list, get the edge value (rating) for the corresponding user-item pair from the `edge_values` dictionary
        edge_values = [
            self.edge_values[(user.item(), item.item())]
            for user, item in user_item_pairs
        ]

        # Create a tensor from the list of edge values and transpose it
        self.y = torch.tensor(edge_values, dtype=torch.float).t()

    def negative_sampling(self):
        """
        Replace positive items with random/unseen items
        """
        print("KG negative sampling...")
        pos_edge_index_r_nps = [
            (
                edge_index,
                np.ones((edge_index.shape[1], 1)) * self.edge_type_dict[edge_type],
            )
            for edge_type, edge_index in self.edge_index_nps.items()
        ]
        pos_edge_index_trans_np = np.hstack([_[0] for _ in pos_edge_index_r_nps]).T
        pos_r_np = np.vstack([_[1] for _ in pos_edge_index_r_nps])
        neg_t_np = np.random.randint(
            low=0, high=self.num_nodes, size=(pos_edge_index_trans_np.shape[0], 1)
        )
        kg_train_data_np = np.hstack([pos_edge_index_trans_np, neg_t_np, pos_r_np])
        kg_train_data_t = torch.from_numpy(kg_train_data_np).long()
        shuffle_idx = torch.randperm(kg_train_data_t.shape[0])
        kg_train_data = kg_train_data_t[shuffle_idx]
        kg_train_data_length = kg_train_data.shape[0]

        print("CF negative sampling...")
        pos_edge_index_trans_np = self.edge_index_nps["user2item"].T
        num_interactions = pos_edge_index_trans_np.shape[0]
        if self.cf_loss_type == "BCE":
            pos_samples_np = np.hstack(
                [
                    pos_edge_index_trans_np,
                    np.ones((pos_edge_index_trans_np.shape[0], 1)),
                ]
            )
            if self.sampling_strategy == "random":
                neg_samples_np = np.hstack(
                    [
                        np.repeat(
                            pos_samples_np[:, 0].reshape(-1, 1),
                            repeats=self.num_negative_samples,
                            axis=0,
                        ),
                        np.random.randint(
                            low=self.type_accs["iid"],
                            high=self.type_accs["iid"] + self.num_iids,
                            size=(num_interactions * self.num_negative_samples, 1),
                        ),
                        torch.zeros((num_interactions * self.num_negative_samples, 1)),
                    ]
                )
            elif self.sampling_strategy == "unseen":
                neg_inids = []
                u_nids = pos_samples_np[:, 0]
                p_bar = tqdm.tqdm(u_nids)
                for u_nid in p_bar:
                    negative_inids = (
                        self.test_pos_unid_inid_map.get(u_nid, [])
                        + self.neg_unid_inid_map[u_nid]
                    )
                    negative_inids = np.random.choice(
                        negative_inids, size=(self.num_negative_samples, 1)
                    )
                    neg_inids.append(negative_inids)
                neg_samples_np = np.hstack(
                    [
                        np.repeat(
                            pos_samples_np[:, 0].reshape(-1, 1),
                            repeats=self.num_negative_samples,
                            axis=0,
                        ),
                        np.vstack(neg_inids),
                        np.zeros((num_interactions * self.num_negative_samples, 1)),
                    ]
                )
            else:
                raise NotImplementedError
            cf_train_data_np = np.vstack([pos_samples_np, neg_samples_np])
        elif self.cf_loss_type == "BPR":
            train_data_np = np.repeat(
                pos_edge_index_trans_np, repeats=self.num_negative_samples, axis=0
            )
            if self.sampling_strategy == "random":
                neg_inid_np = np.random.randint(
                    low=self.type_accs["iid"],
                    high=self.type_accs["iid"] + self.num_iids,
                    size=(num_interactions * self.num_negative_samples, 1),
                )
            elif self.sampling_strategy == "unseen":
                neg_inids = []
                u_nids = pos_edge_index_trans_np[:, 0]
                p_bar = tqdm.tqdm(u_nids)
                for u_nid in p_bar:
                    negative_inids = (
                        self.test_pos_unid_inid_map.get(u_nid, [])
                        + self.neg_unid_inid_map[u_nid]
                    )
                    negative_inids = rd.choices(
                        negative_inids, k=self.num_negative_samples
                    )
                    negative_inids = np.array(negative_inids, dtype=np.long).reshape(
                        -1, 1
                    )
                    neg_inids.append(negative_inids)
                neg_inid_np = np.vstack(neg_inids)
            else:
                raise NotImplementedError
            cf_train_data_np = np.hstack([train_data_np, neg_inid_np])
            if self.entity_aware and not hasattr(self, "iid_feat_nids"):
                # add entity aware data to batches
                movies = pd.read_csv(
                    join(self.processed_dir, "movies.csv"), sep=";"
                ).fillna("")
                tagging = pd.read_csv(join(self.processed_dir, "tagging.csv"), sep=";")
                if self.name == "25m":
                    genome_tagging = pd.read_csv(
                        join(self.processed_dir, "genome_tagging.csv"), sep=";"
                    )

                # Build item entity
                iid_feat_nids = []
                pbar = tqdm.tqdm(self.unique_iids, total=len(self.unique_iids))
                for iid in pbar:
                    pbar.set_description("Sampling item entities...")

                    feat_nids = []

                    year_nid = self.e2nid_dict["year"][
                        movies[movies.iid == iid]["year"].item()
                    ]
                    feat_nids.append(year_nid)

                    genre_nids = [
                        self.e2nid_dict["genre"][genre]
                        for genre in self.unique_genres
                        if movies[movies.iid == iid][genre].item()
                    ]
                    feat_nids += genre_nids

                    actor_nids = [
                        self.e2nid_dict["actor"][actor]
                        for actor in movies[movies.iid == iid]["actors"]
                        .item()
                        .split(",")
                        if actor != ""
                    ]
                    feat_nids += actor_nids

                    director_nids = [
                        self.e2nid_dict["director"][director]
                        for director in movies[movies.iid == iid]["directors"]
                        .item()
                        .split(",")
                        if director != ""
                    ]
                    feat_nids += director_nids

                    writer_nids = [
                        self.e2nid_dict["writer"][writer]
                        for writer in movies[movies.iid == iid]["writers"]
                        .item()
                        .split(",")
                        if writer != ""
                    ]
                    feat_nids += writer_nids

                    tag_nids = [
                        self.e2nid_dict["tid"][tid]
                        for tid in tagging[tagging.iid == iid].tid
                    ]
                    feat_nids += tag_nids
                    if self.name == "25m":
                        genome_tag_nids = [
                            self.e2nid_dict["genome_tid"][genome_tid]
                            for genome_tid in genome_tagging[
                                genome_tagging.iid == iid
                            ].genome_tid
                        ]
                        feat_nids += genome_tag_nids
                    iid_feat_nids.append(feat_nids)
                self.iid_feat_nids = iid_feat_nids

                # Build user entity
                uid_feat_nids = []
                pbar = tqdm.tqdm(self.unique_uids, total=len(self.unique_uids))
                for uid in pbar:
                    pbar.set_description("Sampling user entities...")
                    feat_nids = []

                    tag_nids = [
                        self.e2nid_dict["tid"][tid]
                        for tid in tagging[tagging.uid == uid].tid
                    ]
                    feat_nids += tag_nids
                    uid_feat_nids.append(feat_nids)
                self.uid_feat_nids = uid_feat_nids
        else:
            raise NotImplementedError
        cf_train_data_t = torch.from_numpy(cf_train_data_np).long()
        shuffle_idx = torch.randperm(cf_train_data_t.shape[0])
        cf_train_data = cf_train_data_t[shuffle_idx]
        cf_train_data_length = cf_train_data.shape[0]

        self.train_data_length = min(kg_train_data_length, cf_train_data_length)
        self.train_data = torch.cat(
            [
                cf_train_data[: self.train_data_length],
                kg_train_data[: self.train_data_length],
            ],
            dim=1,
        )

    def __len__(self):
        return len(self.train_data)
        # return self.train_data_length

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, str):
            return getattr(self, idx, None)
        else:
            # dataset[0] == datset.__getitem__(0)
            idx = idx.to_list() if torch.is_tensor(idx) else idx

            train_data_t = self.train_data[idx]
            y = self.y[idx]

            if self.entity_aware:
                inid = train_data_t[1].cpu().detach().item()
                feat_nids = self.iid_feat_nids[int(inid - self.type_accs["iid"])]

                if len(feat_nids) == 0:
                    pos_item_entity_nid = 0
                    neg_item_entity_nid = 0
                    item_entity_mask = 0
                else:
                    pos_item_entity_nid = rd.choice(feat_nids)
                    entity_type = self.nid2e_dict[pos_item_entity_nid][0]
                    lower_bound = self.type_accs.get(entity_type)
                    upper_bound = lower_bound + getattr(
                        self, "num_" + entity_type + "s"
                    )
                    neg_item_entity_nid = rd.choice(range(lower_bound, upper_bound))
                    item_entity_mask = 1

                uid = train_data_t[0].cpu().detach().item()
                feat_nids = self.uid_feat_nids[int(uid - self.type_accs["uid"])]
                if len(feat_nids) == 0:
                    pos_user_entity_nid = 0
                    neg_user_entity_nid = 0
                    user_entity_mask = 0
                else:
                    pos_user_entity_nid = rd.choice(feat_nids)
                    entity_type = self.nid2e_dict[pos_user_entity_nid][0]
                    lower_bound = self.type_accs.get(entity_type)
                    upper_bound = lower_bound + getattr(
                        self, "num_" + entity_type + "s"
                    )
                    neg_user_entity_nid = rd.choice(range(lower_bound, upper_bound))
                    user_entity_mask = 1

                pos_neg_entities = torch.tensor(
                    [
                        pos_item_entity_nid,
                        neg_item_entity_nid,
                        item_entity_mask,
                        pos_user_entity_nid,
                        neg_user_entity_nid,
                        user_entity_mask,
                    ],
                    dtype=torch.long,
                )

                train_data_t = torch.cat([train_data_t, pos_neg_entities], dim=-1)

            return train_data_t, y

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        if isinstance(key, str):
            setattr(self, key, value)
        else:
            raise NotImplementedError("Assignment can't be done outside of constructor")

    def __repr__(self):
        return "{}-{}".format(self.__class__.__name__, self.name.capitalize())
