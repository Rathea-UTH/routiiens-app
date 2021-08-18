#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:06:50 2021

@author: romainloirs
"""

def get_ML_dataset_prediction(pt_depart,road_id,route_data,package_data,stop_data):
    import collections
    from scipy.spatial import Delaunay, ConvexHull, convex_hull_plot_2d
    import numpy as np
    import pandas as pd
    import collections
    import features_creation
    import select_routes
    from utils import compute_distance, get_bearing

    def get_adj_matrix(road_id):
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
    
    def get_zi_stop(road_id, stop):
        res=route_data.loc[road_id,"stops"]
        return res[stop]["zone_id"]
    
    def get_stop_ordonne(road_id):
        res=route_data.loc[road_id,"stops"]
        stop_ordonne=[]
        for k, val in res.items(): 
            stop_ordonne.append(k)
        return stop_ordonne
    
    def get_all_stop_in_zi(road_id,zi):
        dict_stop=route_data.loc[road_id,"stops"]
        res=[]
        for k, val in dict_stop.items(): 
            if dict_stop[k]["zone_id"]==zi:
                res.append(k)
        return res

    def get_all_ZI(road_id):
        stop_ordonne= get_stop_ordonne(road_id)
        res=[]
        for stop in stop_ordonne:
            cluster= get_zi_stop(road_id,stop)
            if cluster not in res:
                res.append(cluster)
        return res
    
    def get_pt_depart_actual_sequence(road_id):
        return pt_depart
    
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
    
    def get_hulls(data_route, route, alpha, depot):
        #stops = data_route[route]['stops']
        #stops = list(set(depot)-{depot})

        stops = data_route.loc[route,'stops']
        stops = list(stops.keys())
        
        pts, zones = [], []
        for stop in stops:
            zones.append(data_route.loc[route,'stops'][stop]['zone_id'])
            pts.append([features_creation.get_lat(data_route, route, stop), features_creation.get_lng(data_route, route, stop)])
        
        points = np.array(pts)
        
        convex_hull = get_convex_hull(pts)
        convex_nodes = get_nodeID(convex_hull.simplices)
        convex_zoneID = zoneID_in_hull(convex_nodes, zones)
        
        
        concave_edges = alpha_shape(points, alpha=alpha, only_outer=True)
        concave_nodes = get_nodeID(concave_edges)
        concave_zoneID = zoneID_in_hull(concave_nodes, zones)
        
        return convex_zoneID, concave_zoneID, points, convex_hull, concave_edges


    def create_sort(indices):
        res = [0]*len(indices)
        for i,e in enumerate(indices):
            res[e] = i
        return res
    
    
    def get_data_zi(road_id,zi,mat):
    
        new_row = {}
    
        vol, nb   = get_volume_colis_zi(road_id,zi)
    
        new_row['Road_id']                = road_id
        new_row["zi_name"]                = zi
        new_row['Km_depot_to_first_zi']   = get_km_first_stop_to_zi(zi,road_id)
        new_row['Tps_depot_to_first_zi']  = distance_zi_pt_depart(road_id, mat,zi)
        new_row['Nb_colis']               = nb 
        new_row['distane_barycentre']     = get_distance_barycentre_to_zi(road_id,zi)
        return new_row


    def get_dataset_Ml(road_i):

        columns=['Road_id',
                 "zi_name",
                 'Tps_depot_to_first_zi',
                 'Nb_colis',
                 'distane_barycentre',
                 'Km_depot_to_first_zi'
                ]
    
        df = pd.DataFrame(columns=columns)
    
        mat=get_adj_matrix(road_id)
        all_zi=get_all_ZI(road_id)
        for zi in all_zi:
            if (type(zi)==str):
                new_row = get_data_zi(road_id,zi,mat)
                df = df.append(new_row, ignore_index=True)

        return df

    X = get_dataset_Ml(road_id)

    alpha = 0.004
    road=road_id
    tmp = X[X['Road_id']==road].copy()
    convex_ZI, concave_ZI, points, convex_hull, concave_edges = get_hulls(route_data, road_id, alpha, pt_depart)

    convex_ZI = collections.Counter(convex_ZI)
    concave_ZI = collections.Counter(concave_ZI)

    tmp['convex'] = tmp['zi_name'].map(convex_ZI) 

    temp = np.argsort(tmp['Km_depot_to_first_zi'])
    tmp['Km_depot_to_first_zi_ind'] = create_sort(temp) 
    tmp['Km_depot_to_first_zi_ind_norm'] = tmp['Km_depot_to_first_zi_ind']/len(tmp['Km_depot_to_first_zi_ind']) 

    temp = np.argsort(tmp['distane_barycentre'])
    tmp['distane_barycentre_ind'] = create_sort(temp)
    tmp['distane_barycentre_ind_norm'] = tmp['distane_barycentre_ind']/len(tmp['distane_barycentre_ind']) 

    
    res={}
    all_zi=get_all_ZI(road_id)
    for zi in all_zi:
        if (type(zi)==str):
            res[zi]={}
            aux=get_all_stop_in_zi(road_id,zi)
            res[zi]=aux
            
    return tmp, res
