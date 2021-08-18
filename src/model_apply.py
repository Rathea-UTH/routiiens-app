#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:00:46 2021

@author: romainloirs
"""
import pandas as pd
import json
import threading
import time as time
from math import ceil
from threading import Thread
from prediction import prediction


path = "data/model_apply_inputs/"

print("***********************")
print("Reading Input Dataset:")
print("***********************")

print("Ouverture de travel_times:")
df_stop_data = json.load(open(path + "new_travel_times.json", "r"))
stop_data = pd.DataFrame.from_dict(df_stop_data, orient="index")
print("Done!")

print("Ouverture de route_data:")
route_data = pd.read_json(path + "new_route_data.json", orient="index")
print("Done!")

print("Ouverture de package_data:")
package_data = pd.read_json(path + "new_package_data.json", orient="index")
print("Done!")

print("Modification de route_data:")
route_data["date_YYYY_MM_DD"] = pd.to_datetime(route_data["date_YYYY_MM_DD"], format = '%Y-%m-%d')
route_data["departure_time_utc"] = route_data["date_YYYY_MM_DD"].astype(str) + route_data["departure_time_utc"]
route_data["departure_time_utc"] = pd.to_datetime(route_data["departure_time_utc"], format = '%Y-%m-%d%H:%M:%S')
print("Done!")

id_route=route_data.index

print("************************")
print("Finding route sequence :")
print("************************")

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
            args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                            **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def to_submission_format(l):
    data = {}
    for dico in l:
        road_id = dico["road_id"]
        data[road_id] = {"proposed":{}}
        sequence_list = dico["Sequence"]
        for i,stop in enumerate(sequence_list):
            data[road_id]["proposed"][stop] = i
    return data

def find_sequence(depart,arrive):
    res_pred=[]

    for i in range(depart, arrive):
        aux=prediction(id_route[i],route_data,package_data,stop_data)
        my_string= "{}: Sequence found in {} seconds for the route n°{}"
        print(my_string.format(id_route[i],round(aux["temps execution"],2),i), flush=True)
        res_pred.append(aux)
    res=to_submission_format(res_pred)
    return res

n = route_data.shape[0]
save_path = "data/model_apply_outputs/"

with open(save_path + "proposed_sequences.json", "w") as tf:
    
    t1 = ThreadWithReturnValue(target=find_sequence, args=(0,ceil(n/10),))
    t2 = ThreadWithReturnValue(target=find_sequence, args=(ceil(n/10),ceil(2*n/10),))
    t3 = ThreadWithReturnValue(target=find_sequence, args=(ceil(2*n/10),ceil(3*n/10),))
    t4 = ThreadWithReturnValue(target=find_sequence, args=(ceil(3*n/10),ceil(4*n/10),))
    t5 = ThreadWithReturnValue(target=find_sequence, args=(ceil(4*n/10),ceil(5*n/10),))
    t6 = ThreadWithReturnValue(target=find_sequence, args=(ceil(5*n/10),ceil(6*n/10),))
    t7 = ThreadWithReturnValue(target=find_sequence, args=(ceil(6*n/10),ceil(7*n/10),))
    t8 = ThreadWithReturnValue(target=find_sequence, args=(ceil(7*n/10),ceil(8*n/10),))
    t9 = ThreadWithReturnValue(target=find_sequence, args=(ceil(8*n/10),ceil(9*n/10),))
    t10 = ThreadWithReturnValue(target=find_sequence, args=(ceil(9*n/10),n,))
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t10.start()

    b1 = t1.join()
    b2 = t2.join() 
    b3 = t3.join() 
    b4 = t4.join() 
    b5 = t5.join() 
    b6 = t6.join() 
    b7 = t7.join() 
    b8 = t8.join() 
    b9 = t9.join() 
    b10 = t10.join() 

    result={}
    result.update(b1)
    result.update(b2)
    result.update(b3)
    result.update(b4)
    result.update(b5)
    result.update(b6)
    result.update(b7)
    result.update(b8)
    result.update(b9)
    result.update(b10)

    json.dump(result,tf)

tf.close()



            









    
    
        
        




