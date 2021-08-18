#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:29:23 2021

@author: ratheauth
"""

import pandas as pd
from scipy.spatial import Delaunay, ConvexHull, convex_hull_plot_2d
import numpy as np
import collections
import json
import ijson
import features_creation
import pickle


path = "data/model_build_inputs/"

print("Importation de package_data...")
data_package = json.load(open(path + "package_data.json", "rt"))
package_data = pd.DataFrame.from_dict(data_package, orient="index")
print("Ok")

print("Importation de route_data...")
data_route = json.load(open(path + "route_data.json", "rt"))
route_data = pd.DataFrame.from_dict(data_route, orient="index")
print("Ok")

high_routes = route_data[route_data["route_score"] == "High"]
high_road_ids = list(high_routes.index)


# ------------------------------------------------------------------------------------------
# Construction d'une nouvelle version de actual sequences, avec des listes de séquences
# au lieu de dict

print("Importation de actual_sequence...")
data_actual_sequences = json.load(open(path + "actual_sequences.json", "r"))
actual_sequence = pd.DataFrame.from_dict(data_actual_sequences, orient="index")
print("Ok")

print("Modification de route_data:")
route_data["date_YYYY_MM_DD"] = pd.to_datetime(route_data["date_YYYY_MM_DD"], format = '%Y-%m-%d')
route_data["departure_time_utc"] = route_data["date_YYYY_MM_DD"].astype(str) + route_data["departure_time_utc"]
route_data["departure_time_utc"] = pd.to_datetime(route_data["departure_time_utc"], format = '%Y-%m-%d%H:%M:%S')
print("Done!")

# Transformation des dictionnaires en séquence (liste) de noms de stops
def actual_sequence_list(route_id):
    l = []
    y_ = actual_sequence.loc[route_id,"actual"]
    n = len(actual_sequence)
    for i in range(n):
        for stop_name, stop_order in y_.items():
            if stop_order == i:
                l.append(stop_name)
                break
    return l

print("transformation de actual_sequences pour avoir des listes...")
seqs = []
for i in range(route_data.shape[0]):
    seqs.append(actual_sequence_list(str(actual_sequence.index[i])))

actuals = pd.DataFrame(data={"actual": seqs}, index=actual_sequence.index)
print("transformation terminée !")


# ------------------------------------------------------------------------------------------

def get_adj_matrix(road_id):

    with open(path + 'travel_times.json', 'r') as f:
        objects = ijson.items(f, road_id, use_float=True)
        columns = list(objects)
    mat_transit = pd.DataFrame.from_dict(columns[0], orient="index")
    return mat_transit

def get_zi_stop(road_id,stop):
    res=route_data.loc[road_id,"stops"]
    return res[stop]["zone_id"]

def get_stop_ordonne(road_id):
    seq=actual_sequence.loc[road_id,"actual"]
    seq_ordonne=sorted(seq.items(), key=lambda t: t[1])
    stop_ordonne=[i[0] for i in seq_ordonne]
    return stop_ordonne

def get_all_ZI(road_id):
    stop_ordonne= get_stop_ordonne(road_id)
    res=[]
    for stop in stop_ordonne:
        cluster= get_zi_stop(road_id,stop)
        if cluster not in res:
            res.append(cluster)
    return res

def get_first_ZI(road_id):
    all_zi=get_all_ZI(road_id)
    return all_zi[1]

def get_first_five_ZI(road_id):
    ordonne_all_zi=get_all_ZI(road_id)
    first_zi=get_first_ZI(road_id)
    while ordonne_all_zi[0]!=first_zi and len(ordonne_all_zi)>0:
        del ordonne_all_zi[0]
    return ordonne_all_zi[0:5]

def get_all_stop_in_zi(road_id,zi):
    dict_stop=route_data.loc[road_id,"stops"]
    res=[]
    for k, val in dict_stop.items(): 
        if dict_stop[k]["zone_id"]==zi:
            res.append(k)
    return res

def get_pt_depart_actual_sequence(road_id):
    return get_stop_ordonne(road_id)[0]

def distance_zi_pt_depart(road_id, matrice_adj,zi):
    stop_in_zi=get_all_stop_in_zi(road_id,zi)
    pt_depart=get_pt_depart_actual_sequence(road_id)
    res=[]
    for stop in stop_in_zi:
        res.append(matrice_adj.loc[pt_depart,stop])
    return sum(res)/len(res)


def get_volume_colis(dic_carac):
    dimensions=dic_carac["dimensions"]
    depth_cm=dimensions["depth_cm"]
    height_cm=dimensions["height_cm"]
    width_cm=dimensions["width_cm"]
    res=depth_cm*height_cm*width_cm
    return res
    

def get_volume_stop(road_id,stop):
    dic_colis=package_data.loc[road_id,stop]
    res=[]
    for k, val in dic_colis.items(): 
        volume_colis=get_volume_colis(val)
        res.append(volume_colis)
    return max(res),len(res)
    

def get_volume_colis_zi(road_id,zi):
    list_stops=get_all_stop_in_zi(road_id,zi)
    res=[]
    nombre_colis=0
    for stop in list_stops:
        volume,nb= get_volume_stop(road_id,stop)
        nombre_colis+=nb
        res.append(volume)
    return max(res),nombre_colis


def get_lat_long_stop(road_id,stop):
    pt=route_data.loc[road_id,"stops"][stop]
    return pt["lat"], pt["lng"]

def get_distance_2_stops(road_id,stop1,stop2):
    from math import sqrt
    lat1,long1=get_lat_long_stop(road_id,stop1)
    lat2,long2=get_lat_long_stop(road_id,stop2)
    distance= sqrt( (lat1-lat2)**2 + (long1-long2)**2)
    return distance

def barycentre_pt(road_id,list_pt):
    lat=[]
    long=[]
    for stop in list_pt:
        lat_pt,lng_pt= get_lat_long_stop(road_id,stop)
        lat.append(lat_pt)
        long.append(lng_pt)
    return sum(lat)/len(lat), sum(long)/len(long)

def get_km_first_stop_to_zi(zi,road_id):
    from math import sqrt
    stop_zi=get_all_stop_in_zi(road_id,zi)
    lat1, long1 =barycentre_pt(road_id,stop_zi)
    
    pt_depart=get_pt_depart_actual_sequence(road_id)
    lat2,long2=get_lat_long_stop(road_id,pt_depart)
    
    distance= sqrt( (lat1-lat2)**2 + (long1-long2)**2)
    return distance

def barycentre_road_id(road_id):
    list_zi=get_all_ZI(road_id)
    lat=[]
    lng=[]
    for zi in list_zi:
        if type(zi)!=float:
            stop_zi=get_all_stop_in_zi(road_id,zi)
            lat1, lng1= barycentre_pt(road_id,stop_zi)
            lat.append(lat1)
            lng.append(lng1)
    return sum(lat)/len(lat), sum(lng)/len(lng)


def get_distance_barycentre_to_zi(road_id,zi):
    from math import sqrt
    lat1, lng1= barycentre_road_id(road_id)
    
    stop_zi=get_all_stop_in_zi(road_id,zi)
    lat2, lng2= barycentre_pt(road_id,stop_zi)
    
    distance= sqrt( (lat1-lat2)**2 + (lng1-lng2)**2)
    return distance


def get_sorted_dict(seq):
    return sorted(seq, key=seq.get)

def get_first_zoneID(data_actual_sequences, data_route, route):
    first = get_sorted_dict(data_actual_sequences[route]['actual'])[1]
    return data_route[route]['stops'][first]['zone_id']

def get_depot(data_actual_sequences, data_route, route):
    return get_sorted_dict(data_actual_sequences[route]['actual'])[0]

def get_first_stop(data_actual_sequences, data_route, route):
    return get_sorted_dict(data_actual_sequences[route]['actual'])[1]

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def get_convex_hull(list_points):
    return ConvexHull(list_points)
    
def get_nodeID(edgelist):
    return {item for t in edgelist for item in t}
    
def zoneID_in_hull(list_nodeID, list_zones):
    list_zoneID = []
    for node in list_nodeID:
        list_zoneID.append(list_zones[node])
    return list_zoneID

def get_hulls(data_route, data_actual_sequences, route, alpha):
    stops = data_actual_sequences[route]
    sorted_actual = get_sorted_dict(stops['actual'])

    pts, zones = [], []
    for stop in sorted_actual[1:]:
        zones.append(data_route[route]['stops'][stop]['zone_id'])
        pts.append([features_creation.get_lat2(data_route, route, stop), features_creation.get_lng2(data_route, route, stop)])
       
    points = np.array(pts)
    #pts = [tuple(i) for i in pts]
    
    convex_hull = get_convex_hull(pts)
    convex_nodes = get_nodeID(convex_hull.simplices)
    convex_zoneID = zoneID_in_hull(convex_nodes, zones)
    
    
    concave_edges = alpha_shape(points, alpha=alpha, only_outer=True)
    concave_nodes = get_nodeID(concave_edges)
    concave_zoneID = zoneID_in_hull(concave_nodes, zones)
    
    first_stop = get_first_stop(data_actual_sequences, data_route, route)
    first_zoneID = get_first_zoneID(data_actual_sequences, data_route, route)
    
    in_convex, in_concave = 0, 0
    if first_zoneID in convex_zoneID: 
        in_convex = 1
    if first_zoneID in concave_zoneID: 
        in_concave = 1
    
    return convex_zoneID, concave_zoneID, points, sorted_actual[1:], convex_hull, concave_edges, first_stop, in_convex, in_concave


def create_sort(indices):
    res = [0]*len(indices)
    for i,e in enumerate(indices):
        res[e] = i
    return res


def get_dataset_ML_firstZi():
    
    def get_data_zi(road_id,zi,mat,my_bool,bool_5):
    
        new_row = {}
        
        vol, nb   = get_volume_colis_zi(road_id,zi)
    
        new_row['Road_id']                = road_id
        new_row["zi_name"]                = zi
        new_row['Km_depot_to_first_zi']   = get_km_first_stop_to_zi(zi,road_id)
        new_row['Tps_depot_to_first_zi']  = distance_zi_pt_depart(road_id, mat,zi)
        new_row['distane_barycentre']     = get_distance_barycentre_to_zi(road_id,zi)
        new_row['Nb_colis']               = nb
        new_row['Is_first_zi']            = my_bool
        new_row['Is_five_first_zi']       = bool_5
    
        return new_row


    def get_dataset_Ml():

        columns=['Road_id',
                 "zi_name",
                 'Tps_depot_to_first_zi',
                 'Km_depot_to_first_zi',
                 'Nb_colis',
                 'distane_barycentre',
                 "Is_first_zi",
                 "Is_five_first_zi",
                ]
    
        df = pd.DataFrame(columns=columns)
        a = 1000
        b = 1100
        for road_id in high_road_ids[a:b]:
        
            mat=get_adj_matrix(road_id)
            first_zi=get_first_ZI(road_id)
            first_five_zi= get_first_five_ZI(road_id)
        
            if type(first_zi)!=float:
                new_row = get_data_zi(road_id,first_zi,mat,1,1)
                df = df.append(new_row, ignore_index=True)
        
                all_zi=get_all_ZI(road_id)
                for zi in all_zi:
                    if (type(zi)!=float and zi!=first_zi):
                    
                        my_bool=0
                        if zi in first_five_zi:
                            my_bool=1
                    
                        new_row = get_data_zi(road_id,zi,mat,0,my_bool)
                        df = df.append(new_row, ignore_index=True)

        return df
    
    # ajouter des colonnes
    newX = pd.DataFrame()
    X = get_dataset_Ml()
    alpha = 0.004
    roads = X['Road_id'].unique()
    for road in roads:
        tmp = X[X['Road_id']==road].copy()
        convex_ZI, concave_ZI, points, sorted_stops, convex_hull, concave_edges, first_stop, in_convex, in_concave = get_hulls(data_route, data_actual_sequences, road, alpha)
    
        convex_ZI = collections.Counter(convex_ZI)
        concave_ZI = collections.Counter(concave_ZI)
    
        tmp['convex'] = tmp['zi_name'].map(convex_ZI)
        tmp['concave'] = tmp['zi_name'].map(concave_ZI)
    
        temp = np.argsort(tmp['Km_depot_to_first_zi'])
        tmp['Km_depot_to_first_zi_ind'] = create_sort(temp)
        tmp['Km_depot_to_first_zi_ind_norm'] = tmp['Km_depot_to_first_zi_ind']/len(tmp['Km_depot_to_first_zi_ind'])

        temp = np.argsort(tmp['distane_barycentre'])
        tmp['distane_barycentre_ind'] = create_sort(temp)
        tmp['distane_barycentre_ind_norm'] = tmp['distane_barycentre_ind']/len(tmp['distane_barycentre_ind'])
    
        newX = pd.concat([newX, tmp])
        
    return newX

print("Création de la dataset d'entraînement 1 pour trouver le 1er zone_id...")
X_first = get_dataset_ML_firstZi()
print("Dataset construite !")


# ------------------------------------------------------------------------------------------
# Construction du modèle de ML pour la datset 1

def get_scaler_and_model_first(X):
    from sklearn.linear_model import LogisticRegression
    from sklearn import preprocessing

    X.dropna(axis=0,inplace=True)
    y = X["Is_first_zi"]
    y = y.astype('int')

    X = X[["Tps_depot_to_first_zi", "Km_depot_to_first_zi_ind_norm", "distane_barycentre_ind_norm", "convex", "Nb_colis"]]

    scaler = preprocessing.StandardScaler().fit(X)
    X_t = scaler.transform(X)

    final_model = LogisticRegression(penalty="l2", C=0.01, solver="liblinear", class_weight="balanced")
    final_model.fit(X_t, y)
    return scaler, final_model


print("Construction et apprentissage du modèle de ML first zone_id...")
scaler_first, model_first = get_scaler_and_model_first(X_first)
print("Apprentissage terminé !")


# -------------------------------------------------------------------

# Sauvegarde du modèle et du scaler

output_dir = "data/model_build_outputs/"

print("Sauvegarde du modèle de ML et du scaler en cours...")
pickle.dump(scaler_first, open(output_dir + "scaler_first_stop.sav", 'wb'))
pickle.dump(model_first, open(output_dir + "model_first_stop.sav", 'wb'))
print("Sauvegarde terminée!")



# ------------------------------------------------------------------------------------------

# Retourne True si le passage stop1->stop2 est emprunté dans la séquence road_id d'Amazon, False sinon
def is_taken(stop1,stop2, road_id):
    actual = actuals.loc[road_id, "actual"]
    n = len(actual)
    for i in range(n-1):
        if actual[i] == stop1 and actual[i+1] == stop2:
            return True 
    return False

# Retourne la moyenne des n distances les plus courtes effectuables depuis stop
def mean_closest(stop,road_id, adj_mat, n=5):
    idx = list(adj_mat.index).index(stop)
    l = list(adj_mat.iloc[idx,:])
    l.sort()
    l = l[:n]
    return sum(l)/n

def temps_livraison(package_data, stop_name, road_id):
    # Moyenne des temps de livraison de chaque colis en ce stop
    m = 0
    N = 0
    # Dict dont les clés sont les ids des paquets et les valeurs des dictionnaires d'infos
    packages_to_deliver = package_data.loc[road_id, stop_name] 
    # Etrange... dans certains stops, pas de colis à déposer ?
    if len(packages_to_deliver) == 0:
        return 0
    for package_id in packages_to_deliver.keys():
        t = packages_to_deliver[package_id]["planned_service_time_seconds"]
        #t = df_package_infos.loc[package_id, "planned_service_time_seconds"]
        m += t
        N += 1
    return m/N

def create_example_for_ML(data, stop1, stop2, road_id, actuals, route_data, package_data, adj_mat):
    data["distance_time"].append(adj_mat.loc[stop1,stop2])
    data["five_closest1"].append(mean_closest(stop1,road_id, adj_mat))
    data["same_zone"].append(int(route_data.loc[road_id,"stops"][stop1]["zone_id"] == route_data.loc[road_id,"stops"][stop2]["zone_id"]))
    data["planned_time_service2"].append(temps_livraison(package_data, stop2, road_id))
    data["y"].append(int(is_taken(stop1,stop2,road_id)))
    return data

def create_road_id_examples_for_ML_col(road_id, actuals, route_data, package_data, n1=3):
    import random
    
    road_exemples = {"distance_time":[], "five_closest1":[],"same_zone":[], "planned_time_service2":[], "y":[]}
    index = []
    adj_mat = get_adj_matrix(road_id)
    n = adj_mat.shape[0]
    col = random.randint(0,n-1)
    stop_arrive = list(adj_mat.index)[col]

    stops = list(actuals.loc[road_id,"actual"])
    for i in range(1,n//2):
        create_example_for_ML(road_exemples, stops[i],stops[i+1], road_id, actuals, route_data, package_data, adj_mat)
        index.append(stops[i] + "_" + stops[i+1] + "/" + road_id)
        
    for s in stops:
        if not is_taken(s,stop_arrive,road_id):
            create_example_for_ML(road_exemples,s,stop_arrive, road_id, actuals, route_data, package_data, adj_mat)
            index.append(stops[i] + "_" + stops[i+1] + "/" + road_id)
        
    df = pd.DataFrame(data=road_exemples, index=index)
    return df

def create_road_id_examples_for_ML_line(road_id, actuals, route_data, package_data, n1=3):
    import random
    
    road_exemples = {"distance_time":[], "five_closest1":[],"same_zone":[], "planned_time_service2":[], "y":[]}
    index = []
    adj_mat = get_adj_matrix(road_id)
    n = adj_mat.shape[0]
    line = random.randint(0,n-1)
    stop_depart = list(adj_mat.index)[line]

    stops = list(actuals.loc[road_id,"actual"])
    for i in range(1,n//2):
        create_example_for_ML(road_exemples, stops[i],stops[i+1], road_id, actuals,route_data, package_data, adj_mat)
        index.append(stops[i] + "_" + stops[i+1] + "/" + road_id)
        
    for s in stops:
        if not is_taken(stop_depart,s,road_id):
            create_example_for_ML(road_exemples,stop_depart, s, road_id, actuals, route_data, package_data, adj_mat)
            index.append(stops[i] + "_" + stops[i+1] + "/" + road_id)
        
    df = pd.DataFrame(data=road_exemples, index=index)
    return df


def create_dataset_for_ML(actuals, route_data, package_data, a=0,b=10):
    new_data = pd.DataFrame(data={"distance_time":[], "five_closest1":[],"same_zone":[], "planned_time_service2":[], "y":[]}, index=[])
    i = 0
    for road_id in high_road_ids[a:b]:
        if i//2 == 0:
            new_exemples = create_road_id_examples_for_ML_col(road_id, actuals, route_data, package_data)
        else:
            new_exemples = create_road_id_examples_for_ML_line(road_id, actuals, route_data, package_data)
        new_data = pd.concat([new_data,new_exemples], axis=0)
    return new_data




# ------------------------------------------------------------------------------------------
# Construction de la dataset pour le ML

print("Création de la dataset d'entraînement 2 ...")
X = create_dataset_for_ML(actuals, route_data, package_data, a=1200, b=1250)
print("Dataset construite !")
#print(df.shape)



# ------------------------------------------------------------------------------------------
# Construction du modèle de ML

# ---------------------------- ML -----------------------------------
def get_scaler_and_model(X):
    from sklearn.linear_model import LogisticRegression
    from sklearn import preprocessing

    X.dropna(axis=0,inplace=True)
    y = X["y"]

    X = X[["five_closest1","distance_time","same_zone","planned_time_service2"]]

    scaler = preprocessing.StandardScaler().fit(X)
    X_t = scaler.transform(X)

    final_model = LogisticRegression(penalty="l2", C=50.0, solver="liblinear", class_weight="balanced")
    final_model.fit(X_t,y)
    return scaler, final_model

print("Construction et apprentissage du modèle de ML 2...")
scaler, model = get_scaler_and_model(X)
print("Apprentissage terminé!")


# -------------------------------------------------------------------

# Sauvegarde du modèle et du scaler

output_dir = "data/model_build_outputs/"

print("Sauvegarde du modèle de ML et du scaler en cours...")
pickle.dump(scaler, open(output_dir + "scaler.sav", 'wb'))
pickle.dump(model, open(output_dir + "model.sav", 'wb'))
print("Sauvegarde terminée!")

