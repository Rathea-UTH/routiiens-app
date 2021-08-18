import random
import pandas as pd

from utils import compute_distance, get_bearing

def sorted_by_values(d):
    return sorted(d, key=d.get)

def get_lat(data_route, route, stop):
    return data_route.loc[route,'stops'][stop]['lat']

def get_lng(data_route, route, stop):
    return data_route.loc[route,'stops'][stop]['lng']

def get_is_station(data_route, route, stop):
    return (data_route.loc[route,'stops'][stop]['type'] == 'Station')

def get_lat2(data_route, route, stop):
    return data_route[route]['stops'][stop]['lat']

def get_lng2(data_route, route, stop):
    return data_route[route]['stops'][stop]['lng']

def get_is_station2(data_route, route, stop):
    return (data_route[route]['stops'][stop]['type'] == 'Station')

def get_dist(data_route, route, stopA, stopB):
    return compute_distance(
        get_lat(data_route, route, stopA),
        get_lng(data_route, route, stopA),
        get_lat(data_route, route, stopB),
        get_lng(data_route, route, stopB)
    )

def get_dir(data_route, route, stopA, stopB):
    return get_bearing(
        get_lat(data_route, route, stopA),
        get_lng(data_route, route, stopA),
        get_lat(data_route, route, stopB),
        get_lng(data_route, route, stopB)
    )

def get_travel_time(data_travel_times, route, stopA, stopB):
    return data_travel_times[route][stopA][stopB]

def get_is_same_zone(data_route, route, stopA, stopB):
    zone_id_A = data_route[route]['stops'][stopA]
    zone_id_B = data_route[route]['stops'][stopB]
    return (zone_id_A == zone_id_B)

def get_nbr_packages(data_package, route, stop):
    return len(data_package[route][stop].keys())

def get_max_vol_packages(data_package, route, stop):
    vol_max = 0
    for package in data_package[route][stop].keys():
        dim = data_package[route][stop][package]['dimensions']
        vol = dim['depth_cm'] * dim['height_cm'] * dim['width_cm']
        vol_max = max(vol, vol_max)
    return vol_max

def get_max_dim_packages(data_package, route, stop):
    vol_max = 0
    for package in data_package[route][stop].keys():
        dim = data_package[route][stop][package]['dimensions']
        vol = max(max(dim['depth_cm'], dim['height_cm']), dim['width_cm'])
        vol_max = max(vol, vol_max)
    return vol_max

def get_planned_service(data_package, route, stop):
    package_1 = list(data_package[route][stop].keys())[0]
    return data_package[route][stop][package_1]['planned_service_time_seconds']

def get_nbr_total_stops(data_route, route):
    return len(data_route[route]['stops'].keys())

def get_nbr_same_zone_stops(data_route, route, stop):
    zone_id = data_route[route]['stops'][stop]['zone_id']
    nbr_same_zone_stops = 0
    for stop_other in data_route[route]['stops'].keys():
        if stop_other != stop:
            zone_id_other = data_route[route]['stops'][stop_other]['zone_id']
            if zone_id_other == zone_id:
                nbr_same_zone_stops += 1
    return nbr_same_zone_stops

def compute_ordered_stops_by_travel_times(data_travel_times, route, stopA):
    stops = list(data_travel_times[route][stopA].keys())
    travel_times = list(data_travel_times[route][stopA].values())
    ordered_stops = [x for _, x in sorted(zip(travel_times, stops))]

    ordered_stops.remove(stopA)
    
    return ordered_stops

def get_avg_travel_time_k_nearest_by_travel_time(data_travel_times, route, ordered_stops, stopA, k=10):
    avg = 0
    for stop_other in ordered_stops[:k]:
        avg += get_travel_time(data_travel_times, route, stopA, stop_other)
    avg /= k
    return avg

def get_features(data_route, data_travel_times, data_package, route, stopA, stopB, ordered_stops_by_travel_time):
    ordered_stops_by_travel_time_without_B = ordered_stops_by_travel_time.copy()
    ordered_stops_by_travel_time_without_B.remove(stopB)

    # About starting stop
    lat = get_lat(data_route, route, stopA)
    lng = get_lng(data_route, route, stopA)
    is_station = get_is_station(data_route, route, stopA)

    # About the arc
    dist = get_dist(data_route, route, stopA, stopB)
    cosdir, sindir = get_dir(data_route, route, stopA, stopB)
    travel_time = get_travel_time(data_travel_times, route, stopA, stopB)
    is_same_zone = get_is_same_zone(data_route, route, stopA, stopB)

    # About the stop B
    nbr_packages = get_nbr_packages(data_package, route, stopB)
    max_vol_packages = get_max_vol_packages(data_package, route, stopB)
    max_dim_packages = get_max_dim_packages(data_package, route, stopB)

    # Informations about global route
    nbr_total_stops = get_nbr_total_stops(data_route, route)
    nbr_same_zone_stops = get_nbr_same_zone_stops(data_route, route, stopA)

    # Informations about travel time with other stops
    nbr_nearer_travel_time_stops = ordered_stops_by_travel_time.index(stopB)
    travel_time_nearest_stop = get_travel_time(data_travel_times, route, stopA, ordered_stops_by_travel_time_without_B[0])
    travel_time_2nd_nearest_stop = get_travel_time(data_travel_times, route, stopA, ordered_stops_by_travel_time_without_B[1])
    avg_travel_time_k_nearest_by_travel_time = get_avg_travel_time_k_nearest_by_travel_time(data_travel_times, route, ordered_stops_by_travel_time_without_B, stopA)
    
    return {
        'lat' : lat,
        'lng' : lng,
        'is_station' : is_station,
        'dist' : dist,
        'cosdir' : cosdir,
        'sindir' : sindir,
        'travel_time' : travel_time,
        'is_same_zone' : is_same_zone,
        'nbr_packages' : nbr_packages,
        'max_vol_packages' : max_vol_packages,
        'max_dim_packages' : max_dim_packages,
        'nbr_total_stops' : nbr_total_stops,
        'nbr_same_zone_stops' : nbr_same_zone_stops,
        'nbr_nearer_travel_time_stops' : nbr_nearer_travel_time_stops,
        'travel_time_nearest_stop' : travel_time_nearest_stop,
        'travel_time_2nd_nearest_stop' : travel_time_2nd_nearest_stop,
        'avg_travel_time_k_nearest_by_travel_time' : avg_travel_time_k_nearest_by_travel_time
    }

def get_actual_sequence(data_actual_sequences, route):
    stop_by_pos = {v: k for k, v in data_actual_sequences[route]['actual'].items()}
    sequence = [stop_by_pos[pos] for pos in range(max(stop_by_pos.keys())+1)]
    return sequence

def get_dataset(data_route, data_travel_times, data_package, data_actual_sequences, routes):
    df = pd.DataFrame()
    
    total_number_of_routes = len(routes)
    route_number = 0
    print("LOADING (" + str(route_number) + "/" + str(total_number_of_routes) + ")", end="\r")
    for route in routes:
        
        route_number += 1
        
        actual_sequence = get_actual_sequence(data_actual_sequences, route)

        for i in range(len(actual_sequence) - 1):
            stopA = actual_sequence[i]
            ordered_stops_by_travel_time = compute_ordered_stops_by_travel_times(data_travel_times, route, stopA)

            # Get actual stop
            stopB = actual_sequence[i + 1]

            # Get 2 stops in the top 10 nearest from A (which are not B)
            random_nearest_stops = [stopB]
            while stopB in random_nearest_stops:
                random_nearest_stops = random.sample(ordered_stops_by_travel_time[:10], 2)
            random_nearest_stops = list(map(lambda stop : (stop, False), random_nearest_stops))

            # Get 2 stops not in the top 10 nearest from A (which are not B)
            random_farest_stops = [stopB]
            while stopB in random_farest_stops:
                random_farest_stops = random.sample(ordered_stops_by_travel_time[10:], 2)
            random_farest_stops = list(map(lambda stop : (stop, False), random_farest_stops))

            stops = [(stopB, True)] + random_nearest_stops + random_farest_stops

            for stop, target in stops:
                features = get_features(data_route, data_travel_times, data_package, route, stopA, stop, ordered_stops_by_travel_time)
                features['target'] = target
                df = df.append(features, ignore_index=True)   
        
        print("LOADING (" + str(route_number) + "/" + str(total_number_of_routes) + ")", end="\r")
    return df