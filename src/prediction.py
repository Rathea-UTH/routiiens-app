#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:00:46 2021

@author: romainloirs
"""

def find_road(current_cluster,pt_depart,matrice_adj_elag,pt_depart_current,pt_arrive_current,aux_time_windows):
    from RO_main import sequence_zone_id
    from RO_main import sequence_zone_id_2
    from RO_main import sequence_zone_id_3
    from time_windows import get_time_windows_stop

    sous_sequence,res_RO_2, res_RO_3= sequence_zone_id_3(current_cluster,matrice_adj_elag,pt_depart_current,pt_arrive_current,aux_time_windows)
    
    if pt_depart in current_cluster:

        if res_RO_2=="Infeasible":
            #supprimer les contraintes de point d'arrivé
            sous_sequence,res_RO_2, res_RO_3= sequence_zone_id(current_cluster,matrice_adj_elag,pt_depart_current,aux_time_windows)

        if res_RO_2=="Infeasible":
            #supprimer les contraintes temporelles
            aux_time_windows=[None]*100
            sous_sequence,res_RO_2, res_RO_3= sequence_zone_id_3(current_cluster,matrice_adj_elag,pt_depart_current,pt_arrive_current,aux_time_windows)

        if res_RO_2=="Infeasible":
            #supprimer les deux constraintes à la fois
            aux_time_windows=[None]*100
            sous_sequence,res_RO_2, res_RO_3= sequence_zone_id(current_cluster,matrice_adj_elag,pt_depart_current,aux_time_windows)

    else:
        if res_RO_2=="Infeasible":
            #supprimer la constrainte de point de départ et d'arrivé
            sous_sequence,res_RO_2, res_RO_3= sequence_zone_id_2(current_cluster,matrice_adj_elag,aux_time_windows)

        if res_RO_2=="Infeasible":
            #supprimer les contraintes temporelles
            aux_time_windows=[None]*100
            sous_sequence,res_RO_2, res_RO_3= sequence_zone_id_3(current_cluster,matrice_adj_elag,pt_depart_current,pt_arrive_current,aux_time_windows)

        if res_RO_2=="Infeasible":
            #supprimer les deux constraintes à la fois
            aux_time_windows=[None]*100
            sous_sequence,res_RO_2, res_RO_3= sequence_zone_id_2(current_cluster,matrice_adj_elag,aux_time_windows)
    
    return sous_sequence,res_RO_2, res_RO_3


# paramétre: l'id de la route
# return: une séquence ordonnée de stops
def prediction(road_id,route_data,package_data,stop_data):
    
    #importations
    from clustering import belong_to  
    from clustering import cluster_conforme
    from clustering import time_next_cluster
    from clustering import find_pt_arrive_depart
    
    from RO_cluster import ordonne_cluster
    from RO_cluster import get_first_zi_to_visit
    from RO_cluster import ordonne_cluster_2
    
    from time_windows import get_time_windows_stop
    from time_windows import translate_time_windows
    
    from ML import elaguer_coefs_bis
    
    from get_data import get_final_adj_matrix
    from get_data import get_road_time_windows
    from get_data import get_zone_id
    from get_data import get_first_stop

    
    import time 
    import pickle
    
    #les résultats de la fonctions
    Sequence=[]
    time_livraison=0
    optimalite=[]

    # Importations pour le ML
    model_path = "data/model_build_outputs/"
    scaler = pickle.load(open(model_path + "scaler.sav", 'rb'))
    model = pickle.load(open(model_path + "model.sav", 'rb'))

    tmps1=time.time()
    
    #obtenir les données du probème 
    matrice_adj=get_final_adj_matrix(road_id,package_data,stop_data)
    all_time_windows=get_road_time_windows(road_id,route_data,package_data,stop_data)
    list_zone_id= get_zone_id(road_id,route_data)
    pt_depart=get_first_stop(road_id,route_data)
      
    #premiere partie: trouver l'ordonnencement des zone_id
    cluster_depart= belong_to(pt_depart, list_zone_id)
    
    first_zi_to_visit= get_first_zi_to_visit(pt_depart,cluster_depart,matrice_adj,list_zone_id,road_id,route_data,package_data,stop_data)

    list_zone_id_ordonne = ordonne_cluster_2(matrice_adj,list_zone_id,cluster_depart,first_zi_to_visit, all_time_windows,15)
    #list_zone_id_ordonne=ordonne_cluster(matrice_adj,list_zone_id,cluster_depart, all_time_windows,15)
   
    # seconde partie : trouver l'ordonnancement au sein des zone_id
    list_cluster=cluster_conforme(matrice_adj,list_zone_id_ordonne,pt_depart,15)
    
    pt_depart_current=pt_depart
    list_depart=[pt_depart]
    
    res_test= []
    
    for current_cluster in list_cluster:
       
        n=len(current_cluster)
        pt_arrive_current,pt_depart_next= find_pt_arrive_depart(pt_depart_current,current_cluster,list_cluster,matrice_adj)
    
        # si le cluster contient 1 stop
        if n==1: 
            Sequence.extend(current_cluster)
            
        #si le cluster contient  plus de 1 stop
        else:
            
            ss_adj_matrice=matrice_adj.loc[current_cluster,current_cluster]
            matrice_adj_elag,n_elag=elaguer_coefs_bis(road_id,ss_adj_matrice,model,scaler,stop_data,route_data,package_data)
            time_windows=get_time_windows_stop(current_cluster,all_time_windows)
            aux_time_windows=translate_time_windows(time_windows, time_livraison)
            
            n_coeff= len(current_cluster)*len(current_cluster) - n_elag
            tmps3=time.time()
            sous_sequence,res_RO_2, res_RO_3= find_road(current_cluster,pt_depart,matrice_adj_elag,pt_depart_current,pt_arrive_current,aux_time_windows)
            tmps4=time.time()-tmps3
            
            res_test.append([tmps4,n_coeff])
            
            Sequence.extend(sous_sequence)
            optimalite.append(res_RO_2)
            time_livraison+=res_RO_3
            
        pt_depart_current=pt_depart_next
        list_depart.append(pt_depart_current)
        
        aux= time_next_cluster(current_cluster,list_cluster,matrice_adj)
        time_livraison+=aux
        
    tmps2=time.time()-tmps1
    
    is_valid=True
    if Sequence[0]!=pt_depart:
        is_valid=False
    
    n_stops= sum([len(zone_id) for zone_id in list_zone_id ])
    
    if n_stops!=len(set(Sequence)):
        is_valid=False
    
    res={}
    res["road_id"]=road_id
    res["n_cluster"]=len(list_zone_id)
    res["n_stops"]=n_stops
    res["pt_depart"]=pt_depart
    res["temps de livraison"]=time_livraison
    res["temps execution"]=tmps2
    res["Sequence"]=Sequence
    res["zone_id"]=list_zone_id
    res["list_cluster"]=list_cluster
    res["test_elagage"]=res_test
            
    return res