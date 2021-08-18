from math import sin, cos, sqrt, atan2, radians, sin, cos
from turfpy import measurement
from geojson import Point, Feature

def compute_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0
    
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def get_bearing(lat1, lng1, lat2, lng2):
    start = Feature(geometry=Point((lat1, lng1)))
    end = Feature(geometry=Point((lat2, lng2)))
    bearing_degree = measurement.bearing(start,end)
    bearing_radians = radians(bearing_degree)
    return cos(bearing_radians), sin(bearing_radians)