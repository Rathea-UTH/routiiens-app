#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:41:34 2021

@author: romainloirs
@content: Fonctions de recherche du plus court chemin
"""

def RO_main(P,indice_depart,indice_arrive,time_windows):
    
    #importation
    import pulp
    import numpy as np
    from min_spanning_tree import kruskal_algo
    
    res_kruskal=kruskal_algo(P)
    
    #####################################
    # Première partie: programme linéaire
    ######################################
    
    #le nombre de sommets du problème
    n_stop=P.shape[0]
    
    #définition du modéle et des variables

    model=pulp.LpProblem('tsp',pulp.LpMinimize)

    x=pulp.LpVariable.dicts("x",((i,j) for i in range(n_stop) \
                                  for j in range(n_stop)),\
                           cat='Binary')
    t=pulp.LpVariable.dicts("t", (i for i in range(n_stop)), \
                             lowBound=0,upBound= np.sum(P), cat='Continuous')
        
    #fonction objectif
    model+=pulp.lpSum(P[i][j]* x[i,j] for i in range(n_stop) \
                      for j in range(n_stop))
    
    # st constraints
    model+=pulp.lpSum(P[i][j]* x[i,j] for i in range(n_stop) for j in range(n_stop)) >= res_kruskal
    
    
    for i in range(n_stop):
        if i!=indice_arrive:
            model += pulp.lpSum(x[i,j] for j in range(n_stop)) == 1 , "contrainte_1b_" + str(i) 
            
    for j in range(n_stop):
        if j!=indice_depart:
            model += pulp.lpSum(x[i,j] for i in range(n_stop)) == 1 , "contrainte_1c_" + str(j)  
            
            
    for i in range(n_stop):
        aux= pulp.lpSum(x[i,j] for j in range(n_stop)) - pulp.lpSum(x[j,i] for j in range(n_stop))
        if i==indice_depart:
            model += aux == 1 , "contrainte_1d_" + str(i)
        elif i==indice_arrive:
            model += aux == -1 , "contrainte_1e_" + str(i)
        else:
            model += aux == 0 , "contrainte_1f_" + str(i)
            
    for i in range(n_stop):
            model += x[i,i] == 0 , "contrainte_1g_" + str(i) 
    
    for i in range(n_stop):
        for j in range(n_stop):
            if i!=j:
                model+= t[i]+P[i][j]-t[j]<= (np.sum(P)+np.max(P))*(1-x[i,j])
    
    for i in range(n_stop):
        if time_windows[i]!=None:
            model += t[i]>=time_windows[i][0], "contrainte_1i_g_" + str(i) 
            model += t[i]<=time_windows[i][1], "contrainte_1i_d_" + str(i) 
    
    model += pulp.lpSum(x[i,j] for i in range(n_stop) for j in range(n_stop)) == n_stop-1 , "contrainte arbre"
            
    for i in range(n_stop):
       for j in range(n_stop):
            if P[i][j]==10000:
                model += x[i,j] == 0
                
    if n_stop>2:
        model += x[indice_depart,indice_arrive] == 0
        model += x[indice_arrive,indice_depart] == 0
                
    #trouver la solution du modèle
    model.solve(pulp.GLPK_CMD(path = '/usr/local/bin/glpsol', msg=False, options = ["--tmlim", "10"]))
    status = pulp.LpStatus[model.status]
    
    #############################
    #Partie: trouver la séquence
    #############################

    #obtenir des séquences ordonées selon le premier stop
    def get_plan(r0):
        import copy
        r=copy.copy(r0)
        route = []
        while len(r) != 0:
            plan = [r[0]]
            del (r[0])
            l = 0
            while len(plan) > l:
                l = len(plan)
                for i, j in enumerate(r):
                    if plan[-1][1] == j[0]:
                        plan.append(j)
                        del (r[i])
            route.append(plan)
        return(route) 
    
    # séparer en deux: les bouclées sur eux meme et la chaine qui nous interesse
    def get_plan2(r0):
        import copy
        route=get_plan(r0)
        res1=[]
        res2=[]
        for i in range(len(route)):
            if route[i][0][0]==route[i][len(route[i])-1][1]:
                res1.append(route[i])
            else:
                res2.append(route[i])
        return res1,res2 
    
    # obtenir une séquence sous forme de tuple de la route
    def my_seq(r,indice_depart,indice_arrive):
        res=[]
        current=indice_depart
        for j in range(len(r)):
            for i in range(len(r)):
                if r[i][0]==current:
                    res.append(r[i][0])
                    current=r[i][1]
        res.append(current)    
        return res

    if status=="Optimal":  
        route=[(i,j) for i in range(n_stop) for j in range(n_stop) if pulp.value(x[i,j])==1]
        route_plan1,route_plan2=get_plan2(route) 
        
        #bouet de sauvetage au cas ou la solution n'est pas connexe
        while len(route_plan1)>0:
            for i in range(len(route_plan1)):
                    #print(route_plan[i])
                model+=pulp.lpSum(x[route_plan1[i][j][0],route_plan1[i][j][1]]\
                                        for j in range(len(route_plan1[i])))<= len(route_plan1[i])-1
            status=model.solve()
            route = [(i, j) for i in range(n_stop) for j in range(n_stop) if pulp.value(x[i, j]) == 1]
            route_plan1,route_plan2 = get_plan2(route)
        
        my_route_conforme=[]
        for j in route_plan2:
            my_route_conforme.extend(j)
            
        res=my_seq(my_route_conforme,indice_depart,indice_arrive)
        tt_cost=model.objective.value()
        
        #print("*************************")
        #print("borne_inf:{}".format(res_kruskal))
       # print("res:{}".format(tt_cost))
        
        return res,status,tt_cost
    
    elif status=="Not Solved":
        route=[(i,j) for i in range(n_stop) for j in range(n_stop) if pulp.value(x[i,j])!=0]
        route_plan1,route_plan2=get_plan2(route) 
        
        if (len(route_plan1)>0):
            res=[indice_depart]
            for i in range(n_stop):
                if (i!=indice_depart and i!=indice_arrive):
                    res.append(i)
            res.append(indice_arrive)
            tt_cost=0
            return res,"Infeasible",tt_cost
        
        else:
            my_route_conforme=[]
            for j in route_plan2:
                my_route_conforme.extend(j)
            
            res=my_seq(my_route_conforme,indice_depart,indice_arrive)
            tt_cost=model.objective.value()
            return res,status,tt_cost
    
    else:
        res=[indice_depart]
        for i in range(n_stop):
            if (i!=indice_depart and i!=indice_arrive):
                res.append(i)
        res.append(indice_arrive)
        tt_cost=0
        return res,status,tt_cost
      




# variante de RO_main qui permet de trouver une route sans point d'arrivé
def RO_main_bis(P,indice_depart,time_windows):
    import numpy as np
    n=P.shape[0]
    col=[0]*n
    ligne=[0]*(n+1)
    T=np.insert(P,n,col,axis=1)
    T=np.insert(T,n,ligne,axis=0)
    time_windows.append(None)
    res=RO_main(T,indice_depart,n,time_windows)
    res[0].pop()
    return  res[0],res[1],res[2]


#variante de RO_main qui permet de trouver une route qui est un cycle 
#(le point de départ et d'arrivé coincident)
def RO_main_cycle(P,indice_depart,time_windows):
    import numpy as np
    n=P.shape[0]
    col=P[:,indice_depart]
    ligne=np.insert(P[indice_depart,:],n,[0],axis=0)
    T=np.insert(P,n,col,axis=1)
    T=np.insert(T,n,ligne,axis=0)
    time_windows.append(time_windows[indice_depart])
    res=RO_main(T,indice_depart,n,time_windows)
    res[0].pop()
    return  res[0],res[1],res[2]
    
# variante de RO_main qui permet de trouver une route optimale sans 
# avoir ni point de départ, ni point d'arrivé.
def RO_main_opt(P,time_windows):
    import numpy as np
    n=P.shape[0]
    
    # point de départ fictif
    col=[0]*n
    ligne=[0]*(n+1)
    P=np.insert(P,n,col,axis=1)
    P=np.insert(P,n,ligne,axis=0)
    time_windows.append(None)
    
    #point d'arrivé fictif
    col=[0]*(n+1)
    ligne=[0]*(n+2)
    P=np.insert(P,n+1,col,axis=1)
    P=np.insert(P,n+1,ligne,axis=0)
    time_windows.append(None)
    
    res=RO_main(P,n,n+1,time_windows)
    #supprimer les points fictifs
    res[0].pop()
    del res[0][0]
    return  res[0],res[1],res[2]




def sequence_zone_id(current_cluster,matrice_adj,pt_depart,time_windows):
    n=len(current_cluster)
    P=matrice_adj.to_numpy()
            
    #obtenir l'indice du point de départ
    indice_depart=0
    for i in range(n):
        if current_cluster[i]==pt_depart:
            indice_depart=i   
    
    #faire trouner la RO
    res_RO_1,res_RO_2, res_RO_3= RO_main_bis(P,indice_depart,time_windows)

    # obtenir une séquence de stop à partir des indices
    sous_sequence=[]
    for i in res_RO_1:
        sous_sequence.append(current_cluster[i])
    
    return sous_sequence, res_RO_2,res_RO_3


def sequence_zone_id_2(current_cluster,matrice_adj,time_windows):
    P=matrice_adj.to_numpy()

    #faire trouner la RO
    res_RO_1,res_RO_2, res_RO_3= RO_main_opt(P,time_windows)

    # obtenir une séquence de stop à partir des indices
    sous_sequence=[]
    for i in res_RO_1:
        sous_sequence.append(current_cluster[i])
    
    return sous_sequence, res_RO_2,res_RO_3


def sequence_zone_id_3(current_cluster,matrice_adj,pt_depart,pt_arrive,time_windows):
    
    #obtenir l'indice du point de départ
    n=len(current_cluster)
    indice_depart=0
    indice_arrive=1
    for i in range(n):
        if current_cluster[i]==pt_depart:
            indice_depart=i 
        if current_cluster[i]==pt_arrive:
            indice_arrive=i 
    
    P=matrice_adj.to_numpy()
    
    #faire trouner la RO
    res_RO_1,res_RO_2, res_RO_3= RO_main(P,indice_depart,indice_arrive,time_windows)

    # obtenir une séquence de stop à partir des indices
    sous_sequence=[]
    for i in res_RO_1:
        sous_sequence.append(current_cluster[i])
    
    return sous_sequence, res_RO_2,res_RO_3


def sequence_zone_id_cycle(cluster_zone_id,matrice_cluster,n_cluster_depart,time_windows_2):
    
    for i in range(len(cluster_zone_id)):
        if n_cluster_depart in cluster_zone_id[i]:
            n_cluster_depart_2=i
        
    res=RO_main_cycle(matrice_cluster,n_cluster_depart_2,time_windows_2)
    res2=res[0]

    #récupérer la vrai séquence de zone_id
    seq_cluster=[]
    for i in res2:
        seq_cluster.append(cluster_zone_id[i])
        
    return  seq_cluster






# trace d'anciens travaux

def aux():
    #else:   
        n=n_stop
    
        # point de départ fictif
        col=[0]*n
        ligne=[0]*(n+1)
        P=np.insert(P,n,col,axis=1)
        P=np.insert(P,n,ligne,axis=0)
        time_windows.append(None)
        
        #point d'arrivé fictif
        col=[0]*(n+1)
        ligne=[0]*(n+2)
        P=np.insert(P,n+1,col,axis=1)
        P=np.insert(P,n+1,ligne,axis=0)
        time_windows.append(None)

        new_pt_depart=n
        new_pt_arrive=n+1

        #le nombre de sommets du problème
        n_stop=P.shape[0]
        
        #définition du modéle et des variables

        model_bis=pulp.LpProblem('tsp',pulp.LpMinimize)

        x=pulp.LpVariable.dicts("x",((i,j) for i in range(n_stop) \
                                    for j in range(n_stop)),\
                            cat='Binary')
        t=pulp.LpVariable.dicts("t", (i for i in range(n_stop)), \
                                lowBound=0,upBound= np.sum(P), cat='Continuous')
            
        #fonction objectif
        model_bis+=pulp.lpSum(P[i][j]* x[i,j] for i in range(n_stop) \
                        for j in range(n_stop))
        
        # st constraints
        for i in range(n_stop):
            if i!=new_pt_arrive:
                model_bis += pulp.lpSum(x[i,j] for j in range(n_stop)) == 1 , "contrainte_1b_" + str(i) 
                
        for j in range(n_stop):
            if j!=new_pt_depart:
                model_bis += pulp.lpSum(x[i,j] for i in range(n_stop)) == 1 , "contrainte_1c_" + str(j)  
                
                
        for i in range(n_stop):
            aux= pulp.lpSum(x[i,j] for j in range(n_stop)) - pulp.lpSum(x[j,i] for j in range(n_stop))
            if i==new_pt_depart:
                model_bis += aux == 1 , "contrainte_1d_" + str(i)
            elif i==new_pt_arrive:
                model_bis += aux == -1 , "contrainte_1e_" + str(i)
            else:
                model_bis += aux == 0 , "contrainte_1f_" + str(i)
                
        for i in range(n_stop):
                model_bis += x[i,i] == 0 , "contrainte_1g_" + str(i) 
        
        for i in range(n_stop):
            for j in range(n_stop):
                if i!=j:
                    model_bis+= t[i]+P[i][j]-t[j]<= (np.sum(P)+np.max(P))*(1-x[i,j])
        
        model_bis += pulp.lpSum(x[i,j] for i in range(n_stop) for j in range(n_stop)) == n_stop-1 , "contrainte arbre"
                
        for i in range(n_stop):
            for j in range(n_stop):
                if P[i][j]==10000:
                    model_bis += x[i,j] == 0
                    
        if n_stop>2:
            model += x[new_pt_depart,new_pt_arrive] == 0
            model += x[new_pt_arrive,new_pt_depart] == 0
                    
        
        #trouver la solution du modèle
        #model.solve(solver)
        model_bis.solve(pulp.GLPK_CMD(path = '/usr/local/bin/glpsol', msg=False, options = ["--tmlim", "10"]))
        #model.solve(pulp.PULP_CBC_CMD(maxSeconds=5))
        #model.solve(solver=pulp.GLPK(msg=False))
        status = pulp.LpStatus[model_bis.status]
    
        if status=="Optimal":  
            #print("non optimal résolu")     
            route=[(i,j) for i in range(n_stop) for j in range(n_stop) if pulp.value(x[i,j])==1]
            route_plan1,route_plan2=get_plan2(route) 
            
            #bouet de sauvetage au cas ou la solution n'est pas connexe
            while len(route_plan1)>0:
                for i in range(len(route_plan1)):
                        #print(route_plan[i])
                    model+=pulp.lpSum(x[route_plan1[i][j][0],route_plan1[i][j][1]]\
                                            for j in range(len(route_plan1[i])))<= len(route_plan1[i])-1
                status=model.solve()
                route = [(i, j) for i in range(n_stop) for j in range(n_stop) if pulp.value(x[i, j]) == 1]
                route_plan1,route_plan2 = get_plan2(route)
            
            my_route_conforme=[]
            for j in route_plan2:
                my_route_conforme.extend(j)
            
            #print(my_route_conforme)
            res=my_seq(my_route_conforme,new_pt_depart,new_pt_arrive)
            #print(res)
            #print(new_pt_depart)
            #print(new_pt_arrive)
            del res[0]
            del res[n]
            #print(res)
            tt_cost=model.objective.value()
            return res,status,tt_cost

        else:
            #print("non optimal non résolu")
            res=[indice_depart]
            for i in range(n):
                if (i!=indice_depart and i!=indice_arrive):
                    res.append(i)
            res.append(indice_arrive)
            tt_cost=0
            #print(res)
            return res,status,tt_cost
    
        
    #else:
        #route=[(i,j) for i in range(n_stop) for j in range(n_stop) if pulp.value(x[i,j])==1]
        #route_plan1,route_plan2=get_plan2(route) 
        
        #my_route_conforme=[]
        #for j in route_plan2:
            #my_route_conforme.extend(j)
            
        #res=my_seq(my_route_conforme,indice_depart,indice_arrive)
        #tt_cost=model.objective.value()
        #return res,status,tt_cost




