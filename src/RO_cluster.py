#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:44:07 2021

@author: romainloirs
@content: 
"""


def get_first_zi_to_visit(pt_depart,cluster_depart,matrice_adj,list_zone_id,road_id,route_data,package_data,stop_data):
    
    from ML_first_stop import get_ML_dataset_prediction
    import pickle    
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    #création du dataset
    dataset, dic_zi= get_ML_dataset_prediction(pt_depart,road_id,route_data,package_data,stop_data)
    list_zi=list(dataset["zi_name"])

    to_del = ['Road_id', 'zi_name',"distane_barycentre","distane_barycentre_ind","Km_depot_to_first_zi_ind","Km_depot_to_first_zi"]
    dataset = dataset.drop(to_del, axis = 1)

    # appliquer le modèle de ML
    model_path = "data/model_build_outputs/"
    scaler = pickle.load(open(model_path + "scaler_first_stop.sav", 'rb'))
    model = pickle.load(open(model_path + "model_first_stop.sav", 'rb'))

    dataset = scaler.transform(dataset)
    prediction   = model.predict_proba(dataset)
    
    # Trouver la première zi à l'aide des prédictions.
    prediction_1 = list(prediction[:,1])
    max_value    = max(prediction_1)
    index_max    = prediction_1.index(max_value)
    first_zi= dic_zi[list_zi[index_max]]

    if first_zi in list_zone_id:
        return first_zi 
    else:
        print("la zi n'est pas dans la liste")
        return  first_zi



# obtenir un sous problème compatible avec le programme de RO dans lequel 
# les clusters sont traités comme des stops
# paramétre: la matrice d'adjacence des stops, la list des clusters, le cluster de départ
# et toutes les times windows associés aux stops
# return: la matrice d'adjacence des clusters, le numéro du cluster de départ, 
# les times windows du nouveau problème
def get_matrice_cluster_as_sommet(matrice_adj,list_cluster,cluster_depart,all_time_windows):
        import numpy as np
        from time_windows import get_time_windows_stop
        from time_windows import max_list_time_windows
        from time_windows import min_list_time_windows
        
        n_cluster=len(list_cluster)
        matrice_cluster=np.zeros((n_cluster,n_cluster))
        n_cluster_depart=0
        time_windows=[]
        
        #pour chaque cluster
        for i in range(n_cluster):
            
            #trouver le numéro du cluster de départ
            if list_cluster[i]==cluster_depart:
                n_cluster_depart=i
            
            # déterminer la nouvelle time_windows associé au cluster
            time_windows_stop_cluster=get_time_windows_stop(list_cluster[i],all_time_windows)
            min_time_windows=min_list_time_windows(time_windows_stop_cluster)
            max_time_windows=max_list_time_windows(time_windows_stop_cluster)
            if min_time_windows==max_time_windows:
                time_windows.append(None)
            else:
                time_windows.append((min_time_windows,max_time_windows))
            
            #compléter la ligne de la matrice d'adjacence 
            for j in range(n_cluster):
                if i!=j:
                    matrice_extraite=matrice_adj.loc[list_cluster[i],list_cluster[j]].to_numpy()
                    distance=matrice_extraite.mean()
                    matrice_cluster[i][j]= distance
            
        return matrice_cluster,n_cluster_depart,time_windows

# si un pt est trop éloigné des autres, on le rapproche tout en conservant 
# les proportions pour le problème de RO
def modif_matrice_cluster(matrice_cluster):
        import numpy as np
        vmax=np.nanmax(matrice_cluster)
        ligne_max1=np.where(matrice_cluster == vmax)[0][0]
        ligne_max2=np.where(matrice_cluster == vmax)[1][0]
        
        vmin_1_1=sorted(matrice_cluster[ligne_max1,:].tolist())[1]
        vmin_1_2=sorted(matrice_cluster[:,ligne_max1].tolist())[1]
        
        vmin_2_1=sorted(matrice_cluster[ligne_max2,:].tolist())[1]
        vmin_2_2=sorted(matrice_cluster[:,ligne_max2].tolist())[1]
        
        ligne_max=0
        my_min=0
        if vmin_1_1>vmin_2_1:
            ligne_max=ligne_max1
            if vmin_1_1>vmin_1_2:
                my_min=vmin_1_2
            else: 
                my_min=vmin_1_1
                
        else:
            ligne_max=ligne_max2
            if vmin_2_1>vmin_2_2:
                my_min=vmin_2_2
            else: 
                my_min=vmin_2_1
        
        matrice_cluster[ligne_max,:]=matrice_cluster[ligne_max,:]-my_min
        matrice_cluster[:,ligne_max]=matrice_cluster[:,ligne_max]-my_min
        matrice_cluster[ligne_max,ligne_max]=0


# trouver une route desservant tout les clusters
def ordonne_cluster(matrice_adj,list_cluster, cluster_depart,all_time_windows,seuil=15):
    
    import pandas as pd
    import numpy as np
    from math import ceil
    
    from clustering import same_size_clusters
    from RO_main import RO_main_cycle
    from RO_main import sequence_zone_id
    from RO_main import sequence_zone_id_2
    from RO_main import sequence_zone_id_3
    from RO_main import sequence_zone_id_cycle
    from clustering import find_pt_arrive_depart
    from clustering import cluster_bis
    
    #transformer le problème en un problème pouvant etre résolu par la RO
    matrice_zone_id,n_cluster_depart,time_windows= get_matrice_cluster_as_sommet(matrice_adj,list_cluster,cluster_depart,all_time_windows)
    modif_matrice_cluster(matrice_zone_id)
    n_cluster=matrice_zone_id.shape[0]
 
    # si il y a trop de cluster pour le problème de RO
    if n_cluster>seuil:
                
        #créer des clusters de zone_id
        pd_zone_id=pd.DataFrame(matrice_zone_id)
        cluster_zone_id=cluster_bis(pd_zone_id,seuil)
        
        #trouver une route qui relie les cluster de zone_ID
        matrice_cluster,n_cluster_depart_2,time_windows_2 =get_matrice_cluster_as_sommet(pd_zone_id,cluster_zone_id,n_cluster_depart,time_windows)
        seq_cluster=sequence_zone_id_cycle(cluster_zone_id,matrice_cluster,n_cluster_depart,time_windows_2)
        
        #trouver à l'intérieur des cluster de zone_id, une route qui relie le zone_id
        seq_zone_id=[]
        pt_depart_current=n_cluster_depart
        for zi in seq_cluster:
            pt_arrive_current,pt_depart_next= find_pt_arrive_depart(pt_depart_current,zi,seq_cluster, pd.DataFrame(matrice_zone_id))
            
            if len(zi)==1:
                seq_zone_id+=zi
                
            else:  
                
                list_extraction=np.array([ True if i in zi else False for i in range(n_cluster)],dtype=bool)
                aux=matrice_zone_id[list_extraction,:]
                matrice_zi=aux[:,list_extraction]
                time_windows_zi=[None]*len(zi)
                
                seq,opt,time= sequence_zone_id_3(zi,pd.DataFrame(matrice_zi),pt_depart_current,pt_arrive_current,time_windows_zi)
                seq_zone_id+= seq
                
            pt_depart_current=pt_depart_next
            
        return [list_cluster[i] for i in seq_zone_id]
    
    #si le nombre de cluster est raisonnable
    else:
        res=RO_main_cycle(matrice_zone_id,n_cluster_depart,time_windows)
        res2=res[0]
        return [list_cluster[i] for i in res2 ]









# obtenir un sous problème compatible avec le programme de RO dans lequel 
# les clusters sont traités comme des stops
# paramétre: la matrice d'adjacence des stops, la list des clusters, le cluster de départ
# et toutes les times windows associés aux stops
# return: la matrice d'adjacence des clusters, le numéro du cluster de départ, 
# les times windows du nouveau problème
def get_matrice_cluster_as_sommet_2(matrice_adj,list_cluster,cluster_depart,cluster_arrive,all_time_windows):
        import numpy as np
        from time_windows import get_time_windows_stop
        from time_windows import max_list_time_windows
        from time_windows import min_list_time_windows
        
        n_cluster=len(list_cluster)
        matrice_cluster=np.zeros((n_cluster,n_cluster))
        n_cluster_depart=0
        n_cluster_arrive=0
        time_windows=[]
        
        #pour chaque cluster
        for i in range(n_cluster):
            
            #trouver le numéro du cluster de départ
            if list_cluster[i]==cluster_depart:
                n_cluster_depart=i
            
            if list_cluster[i]==cluster_arrive:
                n_cluster_arrive=i
            
            # déterminer la nouvelle time_windows associé au cluster
            time_windows_stop_cluster=get_time_windows_stop(list_cluster[i],all_time_windows)
            min_time_windows=min_list_time_windows(time_windows_stop_cluster)
            max_time_windows=max_list_time_windows(time_windows_stop_cluster)
            if min_time_windows==max_time_windows:
                time_windows.append(None)
            else:
                time_windows.append((min_time_windows,max_time_windows))
            
            #compléter la ligne de la matrice d'adjacence 
            for j in range(n_cluster):
                if i!=j:
                    matrice_extraite=matrice_adj.loc[list_cluster[i],list_cluster[j]].to_numpy()
                    distance=matrice_extraite.mean()
                    matrice_cluster[i][j]= distance
            
        return matrice_cluster,n_cluster_depart,n_cluster_arrive,time_windows




# trouver une route desservant tout les clusters
def ordonne_cluster_2(matrice_adj,list_cluster, cluster_depart,first_zi_to_visit,all_time_windows,seuil=15):
    
    import pandas as pd
    import numpy as np
    from math import ceil
    
    from clustering import same_size_clusters
    from RO_main import RO_main_cycle
    from RO_main import sequence_zone_id
    from RO_main import sequence_zone_id_2
    from RO_main import sequence_zone_id_3
    from RO_main import sequence_zone_id_cycle
    from clustering import find_pt_arrive_depart
    from clustering import cluster_bis
    from clustering import cluster_ter
    
    #transformer le problème en un problème pouvant etre résolu par la RO
    matrice_zone_id,n_cluster_depart,n_cluster_arrive,time_windows= get_matrice_cluster_as_sommet_2(matrice_adj,list_cluster,first_zi_to_visit,cluster_depart,all_time_windows)
    modif_matrice_cluster(matrice_zone_id)
    n_cluster=matrice_zone_id.shape[0]
 
    # si il y a trop de cluster pour le problème de RO
    if n_cluster>seuil:
        #créer des clusters de zone_id
        pd_zone_id=pd.DataFrame(matrice_zone_id)
        cluster_zone_id=cluster_ter(pd_zone_id,n_cluster_depart,n_cluster_arrive,seuil=15)
        #cluster_zone_id=cluster_bis(pd_zone_id,seuil)

        #trouver une route qui relie les cluster de zone_ID
        #matrice_cluster,n_cluster_depart_2,time_windows_2 =get_matrice_cluster_as_sommet(pd_zone_id,cluster_zone_id,n_cluster_depart,time_windows)
        matrice_zone_id_2,n_cluster_depart_2,n_cluster_arrive_2,time_windows_2= get_matrice_cluster_as_sommet_2(pd_zone_id,cluster_zone_id,n_cluster_depart,n_cluster_arrive,time_windows)
        
        n_cluster_depart_2=0
        n_cluster_arrive_2=0
        for i in range(len(cluster_zone_id)):
            if n_cluster_depart in cluster_zone_id[i]:
                n_cluster_depart_2=i
            if n_cluster_arrive in cluster_zone_id[i]:
                n_cluster_arrive_2=i

        seq_cluster=sequence_zone_id_3(cluster_zone_id,pd.DataFrame(matrice_zone_id_2),cluster_zone_id[n_cluster_depart_2],cluster_zone_id[n_cluster_arrive_2],time_windows_2)[0]

        #print(seq_cluster)
        #trouver à l'intérieur des cluster de zone_id, une route qui relie le zone_id
        seq_zone_id=[]
        pt_depart_current=cluster_zone_id[n_cluster_depart_2]
        for zi in seq_cluster:

            pt_arrive_current,pt_depart_next= find_pt_arrive_depart(pt_depart_current,zi,seq_cluster, pd.DataFrame(matrice_zone_id))

            if len(zi)==1:
                seq_zone_id+=zi
                
            else:  
                
                list_extraction=np.array([ True if i in zi else False for i in range(n_cluster)],dtype=bool)
                aux=matrice_zone_id[list_extraction,:]
                matrice_zi=aux[:,list_extraction]
                time_windows_zi=[None]*len(zi)
                
                seq,opt,time= sequence_zone_id_3(zi,pd.DataFrame(matrice_zi),pt_depart_current,pt_arrive_current,time_windows_zi)
                seq_zone_id+= seq
                
            pt_depart_current=pt_depart_next
        
        cluster_depart = seq_zone_id.pop()
        seq_zone_id.insert(0,cluster_depart)
            
        return [list_cluster[i] for i in seq_zone_id]
    
    #si le nombre de cluster est raisonnable
    else:
        time_windows_zi=[None]*10000000
        res=sequence_zone_id_3(list(range(n_cluster)),pd.DataFrame(matrice_zone_id),n_cluster_depart,n_cluster_arrive,time_windows)
        res2=res[0]
        cluster_depart = res2.pop()
        res2.insert(0,cluster_depart)
        return [list_cluster[i] for i in res2]









