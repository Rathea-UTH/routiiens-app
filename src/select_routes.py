import random
import features_creation

def get_route_score(data_route, route, score):
    return (data_route[route]['route_score'] == score)

def is_in_LA(data_route, route):
    first_stop = list(data_route[route]['stops'].keys())[0]
    lat = features_creation.get_lat(data_route, route, first_stop)
    lng = features_creation.get_lng(data_route, route, first_stop)
    return (31 < lat) & (lat < 39) & (-123 < lng) & (lng < -115)

def get_routes_in_LA_with_score(data_route, score):
    routes = []
    for route in data_route.keys():
        if is_in_LA(data_route, route) & get_route_score(data_route, route, score):
            routes.append(route)
    return routes

def train_test_routes(routes, split=0.8):
    random.shuffle(routes)
    split = int(len(routes) * split)
    return routes[:split], routes[split:]

