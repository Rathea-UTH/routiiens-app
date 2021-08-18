#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:46:09 2021

@author: romainloirs
"""

#extraire les times windows des sommets de la list_stop
def get_time_windows_stop(list_stop,all_time_windows):
        res=[]
        for stops in list_stop:
            res.append(all_time_windows[stops])
        return res

#retourne une time windows translaté de celle placé en arguement 
def translate_time_windows(time_windows, nb):
    my_copy=list(time_windows)
    for i in range(len(my_copy)):
        if my_copy[i]!=None:
            my_copy[i][0]=time_windows[i][0]-nb
            my_copy[i][1]=time_windows[i][1]-nb
    return my_copy

#obtnenir le max des bornes sup d'une liste de time_windows
def max_list_time_windows(time_windows):
        res=-1
        for fenetre in time_windows:
            if fenetre!=None:
                if fenetre[1]>res:
                    res=fenetre[1]
        if res==-1:
            return None
        else: 
            return res
    
#obtnenir le min des bornes inf d'une liste de time_windows
def min_list_time_windows(time_windows):
        res=10e5
        for fenetre in time_windows:
            if fenetre!=None:
                if fenetre[0]<res:
                    res=fenetre[0]
        if res==10e5:
            return None
        else: 
            return res