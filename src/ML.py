#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:06:50 2021

@author: romainloirs
"""


# ---------------------------- ML -----------------------------------

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
    if len(packages_to_deliver) == 0:
        return 0
    for package_id in packages_to_deliver.keys():
        t = packages_to_deliver[package_id]["planned_service_time_seconds"]
        #t = df_package_infos.loc[package_id, "planned_service_time_seconds"]
        m += t
        N += 1
    return m/N


def elaguer_coefs_bis_sort(road_id,ss_adj_matrice,model,scaler,route_data,stop_data,package_data,thres_elag=0.3):
    import numpy as np
    from time import time
    
    def mean_closest(stop,road_id, adj_mat, n=5):
        idx = list(adj_mat.index).index(stop)
        l = list(adj_mat.iloc[idx,:])
        l.sort()
        l = l[:n]
        return sum(l)/n
    
    def create_example(stop1, stop2, road_id, stop_data, route_data, package_data, ss_adj_mat):
        data = []
        data.append(mean_closest(stop1,road_id, ss_adj_mat))
        aux=ss_adj_mat.loc[stop1,stop2]
        while(not isinstance(aux, np.float64)):
            aux=aux.iloc[0]
        data.append(aux)
        data.append(ss_adj_mat.loc[stop1,stop2])
        data.append(int(route_data.loc[road_id,"stops"][stop1]["zone_id"] == route_data.loc[road_id,"stops"][stop2]["zone_id"]))
        data.append(temps_livraison(package_data, stop2, road_id))
        return np.array([data])
    
    def deg(s,ss_adj_mat,stops):
        x = 0
        for s_ in stops:
            if ss_adj_mat.loc[s,s_] != 10000.0:
                x += 1
        # On enlève le cas ss_adj_mat.loc[s,s]
        x = x - 1
        return x
    
    def sum_proba(s,s_):
        df = create_example(s, s_, road_id, stop_data, route_data, package_data, ss_adj_mat)
        df = scaler.transform(df)
        # On considère aussi la proba de l'arc inverse
        df_ = create_example(s_, s, road_id, stop_data, route_data, package_data, ss_adj_mat)
        df_ = scaler.transform(df_)
        return model.predict_proba(df)[:,1][0] + model.predict_proba(df_)[:,1][0]

    # ------------------------------------------------------------------------

    ss_adj_mat = ss_adj_matrice.copy()
    n = ss_adj_mat.shape[0]
    stops = list(ss_adj_mat.columns)
    ordre_arcs = []
    probas = []
    for i in range(n):
        for j in range(i+1,n):
            s = stops[i]
            s_ = stops[j]
            df = create_example(s, s_, road_id, stop_data, route_data, package_data, ss_adj_mat)
            df = scaler.transform(df)
            # On considère aussi la proba de l'arc inverse
            df_ = create_example(s_, s, road_id, stop_data, route_data, package_data, ss_adj_mat)
            df_ = scaler.transform(df_)
            # Liste des arcs sans orientation
            ordre_arcs.append((stops[i],stops[j]))
            probas.append(model.predict_proba(df)[:,1][0] + model.predict_proba(df_)[:,1][0])
    
    #Parcours des arcs dans un ordre quelconque pour n'en privilégier aucun
    indexes_ordered = np.argsort(probas)
    ordre_arcs = [ordre_arcs[k] for k in indexes_ordered]
    #start = time()
    for (s,s_) in ordre_arcs:
        # Si l'arc entre s et s_ n'est pas pertinent (en prenant en compte les 2 sens)
        if sum_proba(s,s_) < thres_elag:
            # On vérifie qu'on peut retirer l'arc sans violer la condition de Dirac
            if deg(s,ss_adj_mat,stops) > n//2 and deg(s_,ss_adj_mat,stops) > n//2:
                # Puis on élague
                ss_adj_mat.loc[s,s_] = 10000.0
                ss_adj_mat.loc[s_,s] = 10000.0
    #print(time() - start)
    return ss_adj_mat


def elaguer_coefs_bis(road_id,ss_adj_matrice,model,scaler,stop_data,route_data, package_data):
    import numpy as np
    
    def create_example(stop1, stop2, road_id, stop_data, route_data, package_data, ss_adj_mat):
        data = []
        data.append(mean_closest(stop1,road_id, ss_adj_matrice))
        
        aux=ss_adj_mat.loc[stop1,stop2]
        while(not isinstance(aux, np.float64)):
            aux=aux.iloc[0]
        data.append(aux)
        data.append(int(route_data.loc[road_id,"stops"][stop1]["zone_id"] == route_data.loc[road_id,"stops"][stop2]["zone_id"]))

        data.append(temps_livraison(package_data, stop2, road_id))
        
        return np.array([data],dtype=object)

    ss_adj_mat = ss_adj_matrice.copy()
    n = ss_adj_mat.shape[0]
    stops = list(ss_adj_mat.columns)
    # On elague chaque ligne 
    for i,s in enumerate(stops):
        #  Nombre de coefs déjà élagués sur la ligne courante
        nb_elag = len([0.0 for coef in ss_adj_mat.iloc[i,:] if coef==10000.0])
        # Nombre de coefs potentiellement encore élagables
        nb_to_elag = n//2 - nb_elag - 2
        indexes_to_check = []
        probas = []
        for j,s_ in enumerate(stops):
            # On ne peut plus élaguer les stops (lignes) déjà parcourus, sinon 
            # on mettrait de nouveaux coefs à 10000
            if (ss_adj_mat.iloc[i,j] != 10000.0 and j > i):
                df = create_example(s, s_, road_id, stop_data, route_data, package_data, ss_adj_mat)
                df = scaler.transform(df)
                
                df_ = create_example(s_, s, road_id, stop_data, route_data, package_data, ss_adj_mat)
                df_ = scaler.transform(df_)
                probas.append(model.predict_proba(df)[:,1][0] + model.predict_proba(df_)[:,1][0])
                
                indexes_to_check.append(j)
        indexes_probas_sorted = np.argsort(probas)
        indexes_to_check_sorted = [indexes_to_check[k] for k in indexes_probas_sorted]
        # De telle sorte à ne conserver que n//2 coefs sur la ligne
        for j in indexes_to_check_sorted[:nb_to_elag]:
            if j!=i:
                # Théorème de Dirac valable que pour des graphes non orientés
                ss_adj_mat.iloc[i,j] = 10000.0
                ss_adj_mat.iloc[j,i] = 10000.0
        #print(len([0 for coef in adj_mat.iloc[0,:] if coef==10000.0]))
    return ss_adj_mat,nb_elag
     
