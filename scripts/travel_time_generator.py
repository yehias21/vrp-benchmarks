import math
import random
from datetime import datetime, time
from city import Map

def normal_distribution(x, mean, std_dev):
    return math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * math.sqrt(2 * math.pi))

def time_factor(current_time):
    morning_peak = normal_distribution(current_time, 480, 90)
    evening_peak = normal_distribution(current_time, 1020, 90)
    return 0.5 + 2 * (morning_peak + evening_peak)

def random_factor(current_time):
    rush_hour_effect = normal_distribution(current_time, 480, 90) + normal_distribution(current_time, 1020, 90)
    mu = 0 + 0.1 * rush_hour_effect
    sigma = 0.3 + 0.2 * rush_hour_effect
    return random.lognormvariate(mu, sigma)

def accident_probability(current_time):
    return 0.05 * normal_distribution(current_time, 1260, 120)  # Peak at 9 PM (21:00)

def accident_delay():
    return random.uniform(0.5, 2)

def calculate_delay(distance, current_time):
    time_fac = time_factor(current_time)
    distance_factor = 1 - math.exp(-distance / 50)
    base_delay = 0.25 * time_fac * distance_factor
    rand_factor = random_factor(current_time)
    delay = base_delay * rand_factor
    
    # Check for accident
    if random.random() < accident_probability(current_time):
        delay += accident_delay()
    
    return delay

def sample_travel_time(a, b, distances, current_time, velocity=1):
    distance = distances[(a, b)]
    delay = calculate_delay(distance, current_time)
    return distance / velocity + delay

def get_distances(map):
    distances = {}
    locations = map.locations
    for i in range(len(locations)):
        for j in range(len(locations)):
            distance = locations[i].distance(locations[j])
            distances[(i, j)] = distance
    return distances

if __name__ == "__main__":
    map = Map((100, 100), 1, 1)
    map.sample_locations(2)
    
    distances = get_distances(map) # calculate euclidean distances between all locations in map.locations
    current_time = 500 # start time of travel in [1, 1440]
    a = 0 # index of departure location in map.locations 
    b = 1 # index of arrival location in map.locations
    travel_time = sample_travel_time(a, b, distances, current_time, velocity=1)
    print(travel_time)
    
