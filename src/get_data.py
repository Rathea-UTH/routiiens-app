#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:57:01 2021

@author: romainloirs
"""


def get_adj_matrix(road_id,stop_data):
    import pandas as pd
    # Tri de stop_data
    ordered_stops = list(stop_data.columns)
    ordered_stops.sort()
    stop_data_bis = stop_data[ordered_stops]
    
    # Construction de la matrice d'adjacence
    entrees = []
    for i,stop in enumerate(ordered_stops):
        # NaN -> float, non NaN -> dict
        if type(stop_data_bis.loc[road_id,stop]) == dict:
            entrees.append(stop_data_bis.columns[i])
    d = {}
    for stop_name in entrees:
        d[stop_name] = list(stop_data_bis[stop_name][road_id].values())
    mat_transit = pd.DataFrame(data=d, index=entrees)
    return mat_transit
 

# obtenir la matrice d'adjacence modifié avec la ML
def get_final_adj_matrix(road_id,package_data,stop_data):
    # Fcts intermédiaires
    
    # package_data n'a pas besoin d'être transformée
    def temps_livraison(stop_name, road_id):
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
    
    def add_deliveries_to_matrix(m, road_id):
        stops = list(m.index)
        for i,stop in enumerate(stops):
            tps_delivery_stop = temps_livraison(stop, road_id)
            m.iloc[i,:] = m.iloc[i,:] + tps_delivery_stop
    # ----------------------------------------------------------------------------
    
    m = get_adj_matrix(road_id,stop_data)
    add_deliveries_to_matrix(m, road_id)
    return m


    # Fcts intermédiaires
    
    # package_data n'a pas besoin d'être transformée
    def temps_livraison(stop_name, road_id):
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
    
    def add_deliveries_to_matrix(m, road_id):
        stops = list(m.index)
        for i,stop in enumerate(stops):
            tps_delivery_stop = temps_livraison(stop, road_id)
            m.iloc[i,:] = m.iloc[i,:] + tps_delivery_stop
    # ----------------------------------------------------------------------------
    
    m = get_adj_matrix(road_id,stop_data)
    add_deliveries_to_matrix(m, road_id)
    return m

# Retourne une liste dont les éléments sont les fenêtres temporelles [a,b] des stops visités par road_id
# (les stops associés sont dans le même ordre que dans la matrice d'adjacence de la route)
def get_road_time_windows(road_id,route_data,package_data,stop_data):
    
    import datetime
    adj_mat = get_adj_matrix(road_id,stop_data)
    
    def stop_time_window(stop_name, road_id):
        # Initialisation de a au plus tôt (ie départ de la tournée)
        a = route_data.loc[road_id,"departure_time_utc"]
        # Initialisation de b au plus tard (ie heure actuelle, sachant que les livraisons ont toutes déjà eu lieu)
        b = datetime.datetime.now()
        b_bis = b
        # Dict dont les clés sont les ids des paquets et les valeurs sont des dictionnaires d'infos
        packages_to_deliver = package_data.loc[road_id, stop_name] 
        for package_id in packages_to_deliver.keys():

            a_package = packages_to_deliver[package_id]["time_window"]["start_time_utc"]
            b_package = packages_to_deliver[package_id]["time_window"]["end_time_utc"]
            if a_package != None:

                a_package = datetime.datetime.strptime(a_package, '%Y-%m-%d %H:%M:%S')
                if a_package > a:
                    a = a_package
            if b_package != None:
                b_package = datetime.datetime.strptime(b_package, '%Y-%m-%d %H:%M:%S')
                if b_package < b:
                    b = b_package  
        # Gestion du cas où absence de contraintes
        if a == route_data.loc[road_id,"departure_time_utc"]:
            a = None
        if b == b_bis:
            b = None
        return [a,b]
    
    def get_ordered_stops(road_id,stop_data):
        ordered_stops = list(stop_data.columns)
        ordered_stops.sort()
        stop_data_bis = stop_data[ordered_stops]

        entrees = []
        for i,stop in enumerate(ordered_stops):
            # NaN -> float, non NaN -> dict
            if type(stop_data_bis.loc[road_id,stop]) == dict:
                entrees.append(stop_data_bis.columns[i])
        return entrees
    
    # ----------------------------------------------------------------------------------------------------
    start = route_data.loc[road_id,"departure_time_utc"]
    stops = get_ordered_stops(road_id,stop_data)
    time_windows = {}
    for stop in stops:
        tmp =  stop_time_window(stop, road_id)
        if tmp == [None,None]:
            time_windows[stop] = None
        else:
            a_stop=tmp[0]
            b_stop=tmp[1]
            a = (a_stop - start).total_seconds() if a_stop != None else 0
            b = (b_stop - start).total_seconds() if b_stop != None else adj_mat.sum().sum()
            time_windows[stop] = [a,b]
    return time_windows

#obtenir les zone_id
def get_zone_id(road_id,route_data):
    stops = route_data.loc[road_id,"stops"]
    d = {"None": []}

    for s in stops.keys():
        if type(stops[s]["zone_id"]) == str:
            if stops[s]["zone_id"] in d.keys():
                d[stops[s]["zone_id"]].append(s)
            else:
                d[stops[s]["zone_id"]] = [s]
        else:
            d["None"].append(s)
    return [d[i] for i in d]
    
#obtenir le stop de départ à partir du dataframe des distances
# le point de départ est le point le plus éloigné des autres
def get_first_stop(road_id,route_data):
    n=route_data.shape[0]
    save_stops = list(route_data["stops"])
    road_names = list(route_data.index)
    index_number = road_names.index(road_id)
    for item in save_stops[index_number]:
        if (save_stops[index_number][item]['type']=='Station'):
            return item
        
        
        
        
        
        