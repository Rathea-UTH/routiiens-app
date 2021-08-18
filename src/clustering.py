#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:41:34 2021

@author: romainloirs
@content: fonction de clustering
"""


# Retourne le cluster auquel appartient le stop 
def belong_to(stop, clusters):
    for cluster in clusters:
        if stop in cluster:
            return cluster


# On prend un stop au hasard, on regarde les 9 stops les plus proches de celui-ci, on en fait un cluster puis
# on les retire de la liste. On recommence: on considère un stop au hasard parmi les stops restants ...
# Daire attention à la fin ! En effet, s'il ne reste que 11 clusters et qu'on applique le procédé ci-dessus, on se
# retrouvera avec un cluster de taille 10 et un cluster de taille 2
# Ainsi, si le nombre restant de stops est compris entre 10 et 15 (inclus), on met tous ces stops dans un même cluster
# Si le nombre restant de stops est compris entre 16 et 20 (inclus), on va appliquer le procédé.
def same_size_clusters(adj_mat,nb_cluster=15):
    import random 
    import numpy as np
    
    stops = list(adj_mat.index)
    clusters = []
    while len(stops) > nb_cluster:
        n = len(stops)
        # choix aléatoire d'un stop parmi les n restants
        chosen_stop = stops[random.randint(0, n-1)]
        # Calcul des distances 
        distances = []
        stops.remove(chosen_stop)
        for stop in stops:
            distances.append(adj_mat.loc[chosen_stop, stop])
        # Indices des stops du plus proche au plus éloigné de chosen_stop
        sorted_indexes = np.argsort(distances)
        # Construction nouveau cluster
        cluster = [stops[i] for i in sorted_indexes[:nb_cluster-1]]
        cluster.append(chosen_stop)
        #print(cluster)
        clusters.append(cluster)
        # On retire les stops du nouveau cluster de la liste des stops
        for stop in cluster:
            # On a déjà retiré chosen_stop de la liste des stops
            if stop != chosen_stop:
                stops.remove(stop)
            
    clusters.append(stops)
    return clusters


# regrouper des zone_id par petit groupes pour avoir un nombre de cluster zone_id inférieur à 15
def cluster_bis(adj_mat,seuil=15):
    
    import numpy as np
    from math import ceil
    P=adj_mat.to_numpy()

    def list_ordonne():
        list_element=np.copy(P).tolist()
        list_el=[]
        for i in list_element:
            list_el.extend(i)
        list_el.sort()
        while(list_el[0]==0):
            del list_el[0]
        return list_el

    n=adj_mat.shape[0]
    list_zi=list(adj_mat.index)
    el_ordonne=list_ordonne()
    res=[]
    n_zone_id_by_cluster= ceil(n/seuil)

    while (n>seuil and len(el_ordonne)!=0):
        min=el_ordonne[0]
        del el_ordonne[0]
        min_indexes = np.where(P == min) 
        zi_1=min_indexes[0][0]
        zi_2=min_indexes[1][0]

        if zi_1 in list_zi and zi_2 in list_zi:
            list_zi.remove(zi_1)
            list_zi.remove(zi_2)
            res.append([zi_1,zi_2])

        elif zi_1 not in list_zi and zi_2 in list_zi:
            i=0
            while zi_1 not in res[i]:
                i+=1
            aux=res[i]

            if len(aux)<n_zone_id_by_cluster:
                aux.append(zi_2)
                del res[i]
                res.append(aux)
                list_zi.remove(zi_2)
        
        elif zi_1 in list_zi and zi_2 not in list_zi:
            i=0
            while zi_2 not in res[i]:
                i+=1
            aux=res[i]
            if len(aux)<n_zone_id_by_cluster:
                aux.append(zi_1)
                del res[i]
                res.append(aux)
                list_zi.remove(zi_1)
        
        else:
            id_1=0
            id_2=0
            for i in range(len(res)):
                if zi_1 in res[i]:
                    id_1=i
                if zi_2 in res[i]:
                    id_2=i
            if id_1!=id_2:
                aux1=res[id_1]
                aux2=res[id_2]
                aux=aux1+ aux2
                if len(aux)<n_zone_id_by_cluster:
                    if id_1>id_2:
                        del res[id_1]
                        del res[id_2]
                    else:
                        del res[id_2]
                        del res[id_1]
                    res.append(aux)
                    
        n=len(res)+len(list_zi)
    
    for zi in list_zi:
        res.append([zi])
        
    return res

# regrouper des zone_id par petit groupes pour avoir un nombre de cluster zone_id inférieur à 15
def cluster_ter(adj_mat,pt_depart,pt_arrive,seuil=15):
    import numpy as np
    from math import ceil
    P=adj_mat.to_numpy()


    def list_ordonne():
        list_element=np.copy(P).tolist()
        list_el=[]
        for i in list_element:
            list_el.extend(i)
        list_el.sort()
        while(list_el[0]==0):
            del list_el[0]
        return list_el

    n=adj_mat.shape[0]
    list_zi=list(adj_mat.index)
    el_ordonne=list_ordonne()
    res=[]
    n_zone_id_by_cluster= ceil(n/seuil)

    while (n>seuil and len(el_ordonne)!=0):
        min=el_ordonne[0]
        del el_ordonne[0]
        min_indexes = np.where(P == min) 
        zi_1=min_indexes[0][0]
        zi_2=min_indexes[1][0]

        condition = (zi_1!=pt_depart and zi_2!=pt_arrive) and (zi_1!=pt_arrive and zi_2!=pt_depart)

        if  condition:

            if zi_1 in list_zi and zi_2 in list_zi:
                list_zi.remove(zi_1)
                list_zi.remove(zi_2)
                res.append([zi_1,zi_2])

            elif zi_1 not in list_zi and zi_2 in list_zi:
                i=0
                while zi_1 not in res[i]:
                    i+=1
                aux=res[i]

                if len(aux)<n_zone_id_by_cluster:
                    aux.append(zi_2)
                    del res[i]
                    res.append(aux)
                    list_zi.remove(zi_2)
            
            elif zi_1 in list_zi and zi_2 not in list_zi:
                i=0
                while zi_2 not in res[i]:
                    i+=1
                aux=res[i]
                if len(aux)<n_zone_id_by_cluster:
                    aux.append(zi_1)
                    del res[i]
                    res.append(aux)
                    list_zi.remove(zi_1)
            
            else:
                id_1=0
                id_2=0
                for i in range(len(res)):
                    if zi_1 in res[i]:
                        id_1=i
                    if zi_2 in res[i]:
                        id_2=i
                if id_1!=id_2:
                    aux1=res[id_1]
                    aux2=res[id_2]
                    aux=aux1+ aux2
                    if len(aux)<n_zone_id_by_cluster:
                        if id_1>id_2:
                            del res[id_1]
                            del res[id_2]
                        else:
                            del res[id_2]
                            del res[id_1]
                        res.append(aux)
                        
            n=len(res)+len(list_zi)
    
    for zi in list_zi:
        res.append([zi])
        
    return res



# la fonction permet de vérifier si la liste des clusters est conformes pour la RO,
# ie chaque cluster ne contient pas plus de 15 stops
# param:  la matrice d'adjacence de tout les stops du problème,la list des clusters,
# le seuil du nombre de stop à ne pas dépasser dans chaque cluster
# return des clusters contenant au plus seuil stops
def cluster_conforme(adj_mat,list_cluster,pt_depart,seuil=15):
    from clustering import same_size_clusters
    
    aux_cluster=[]
    for i in range(len(list_cluster)):
        n=len(list_cluster[i])
        if n>seuil:
            matrice_adjacence=adj_mat.loc[list_cluster[i],list_cluster[i]]
            new_clusters=same_size_clusters(matrice_adjacence,seuil)
            
            for i in range(len(new_clusters)):
                if pt_depart in new_clusters[i]:
                    aux_cluster.append(new_clusters[i])
                    del new_clusters[i]
                    break
                
            aux_cluster.extend(new_clusters)
        else:
            aux_cluster.append(list_cluster[i])
    
    return aux_cluster


def time_next_cluster(current_cluster,list_cluster,matrice_adj):
    n=len(list_cluster)
    for i in range(n):
        if list_cluster[i]==current_cluster:
            n_current_cluster= i
            # c'est le dernier de la liste
            if n_current_cluster== n-1:
                return 0
            else:
                next_cluster=list_cluster[i+1]
                P=matrice_adj.loc[current_cluster,next_cluster].to_numpy()
                return P.mean()  


# trouver les points d'arrivé d'un cluster et le point de départ du cluster suivant
def find_pt_arrive_depart(pt_depart_current,current_cluster,list_cluster,matrice_adj):
    import numpy as np
    import copy as copy
    n=len(list_cluster)
    
    #trouver le cluster suivant
    k=0
    while (list_cluster[k]!=current_cluster):
        k+=1
    #on est sur l'indice du cluster suivant 
    k+=1
    # si on est sur le dernier cluster
    if k==n:
        k=0
    
    next_cluster=list_cluster[k]
    current_cluster_del=copy.copy(current_cluster)
    
    if len(current_cluster)!=1:
        current_cluster_del=copy.copy(current_cluster)
        current_cluster_del.remove(pt_depart_current)
    
    P=matrice_adj.loc[current_cluster_del,next_cluster]
    M= P.to_numpy()
    my_min=np.min(M)
    x_min,y_min=np.where(M==my_min)
    pt_arrive_current=P.index[x_min[0]]
    pt_depart_next=P.columns[y_min[0]]
    return pt_arrive_current,pt_depart_next




