from typing import Tuple, List
import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass
import math
from constants import CUSTOMER, DEPOT
from sklearn.cluster import KMeans

@dataclass
class Location:
    x: float
    y: float
    type: str = CUSTOMER

    def distance(self, other: "Location") -> int:
        return np.linalg.norm([self.x - other.x, self.y - other.y]).astype(int)

class City:
    def __init__(self, center: Tuple[float, float], spread: float):
        self.center = np.array(center)
        self.spread = spread

    def batch_sample(self, map_size: Tuple[int, int], n: int) -> List[Location]:
        locations = []         # List to hold valid sampled points (as tuples)
        point_counts = {}      # Dictionary to count occurrences of each tuple
        max_occurrence: int = 2
        iteration = 0

        while len(locations) < n and iteration < 30:
            remaining = n - len(locations)
            # Sample remaining points
            samples = np.random.normal(self.center, self.spread, size=(remaining, 2))
            # Clip values so they stay within the map bounds, then convert to integers
            samples_clipped = np.clip(samples, [0, 0], map_size).astype(int)
            
            for point in samples_clipped:
                pt = tuple(point)  # convert array to tuple for hashing
                # Only add the point if it has not reached max_occurrence
                if point_counts.get(pt, 0) < max_occurrence:
                    locations.append(pt)
                    point_counts[pt] = point_counts.get(pt, 0) + 1
                    # Stop if we have reached the desired number of points
                    if len(locations) == n:
                        break
            iteration += 1

        if len(locations) < n:
            print(f"Delta (n - current points): {n - len(locations)}")
        
        # Convert each tuple to a Location instance before returning
        return [Location(x, y) for x, y in locations]


    def __repr__(self):
        return f"City(center={self.center}, spread={self.spread})"

class Map:
    def __init__(self, size: Tuple[int, int], num_cities: int, num_depots: int):
        self.size = size
        self.num_cities = num_cities
        self.num_depots = num_depots

        width, height = size
        area = width * height
        # Compute desired spread based on area and number of cities
        # Add a small clamp to avoid zero spread if num_cities is large
        spread = math.sqrt(area / (math.pi * num_cities)) if num_cities > 0 else 1
        self.spread_range = (0.3*spread, 0.4*spread)

        # Step 1: Prepare data points: all integer coordinates in the map
        # This could be large, consider downsampling if needed
        points = [(x, y) for x in range(size[0]) for y in range(size[1])]
        points = np.array(points)

        # Step 2: Run KMeans on these points to find city centers
        kmeans = KMeans(n_clusters=int(num_cities), init='random', n_init=10, max_iter=300)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_

        city_centers = []
        for c in centers:
            x, y = c
            # Round and clamp
            x = int(min(max(round(x), 0), size[0] - 1))
            y = int(min(max(round(y), 0), size[1] - 1))
            spread = np.random.randint(*self.spread_range)
            city_centers.append(City((x, y), spread))

        self.cities = city_centers
        self.depots = []
        self.locations = []

    def sample_locations(self, num_locations: int) -> List[Location]:
        locations = []
        loc_per_city = num_locations // len(self.cities)
        leftover = num_locations % len(self.cities)

        for idx, city in enumerate(self.cities):
            n = loc_per_city + (1 if idx < leftover else 0)
            locations.extend(city.batch_sample(self.size, n))

        self.locations = locations

    def cluster_and_place_depots(self):
        customers = np.array([[loc.x, loc.y] for loc in self.locations if loc.type == CUSTOMER])
        if len(customers) == 0:
            return

        if self.num_depots == 1:
            # No need for clustering, just find the mean position
            mean_x = np.mean(customers[:, 0])
            mean_y = np.mean(customers[:, 1])

            # Clamp to map boundaries
            mean_x = int(min(max(round(mean_x), 0), self.size[0] - 1))
            mean_y = int(min(max(round(mean_y), 0), self.size[1] - 1))

            depot = Location(mean_x, mean_y, DEPOT)
            self.depots = [depot]
            self.locations.insert(0, depot )
        else:
            # Proceed with KMeans for multiple depots
            initial_centers = np.array([c.center for c in self.cities[:self.num_depots]])
            kmeans = KMeans(n_clusters=self.num_depots, init=initial_centers, n_init=1, max_iter=300)
            kmeans.fit(customers)

            depots = []
            for center in kmeans.cluster_centers_:
                x, y = center
                x = int(min(max(round(x), 0), self.size[0] - 1))
                y = int(min(max(round(y), 0), self.size[1] - 1))
                depots.append(Location(x, y, DEPOT))

            self.depots = depots
            self.locations = depots + self.locations


    def __repr__(self):
        return f"Map(size={self.size}, num_cities={len(self.cities)}, num_depots={len(self.depots)}, num_locations={len(self.locations)})"


def draw_circle(img: Image, center, color, size, text=""):
    draw = ImageDraw.Draw(img)
    draw.text((center[0], center[1]), text, fill="black")
    draw.ellipse(
        (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        ),
        fill=color,
    )


def map_drawer(map: Map, img_size=(720, 720)) -> Image:
    img = Image.new("RGB", img_size, "white")

    # Draw depots
    for depot in map.depots:
        dd = (
            np.uint32(depot.x) * img_size[0] // map.size[0],
            np.uint32(depot.y) * img_size[1] // map.size[1],
        )
        draw_circle(img, dd, (0, 0, 255), 15, "D")

    # Draw customers
    for loc in map.locations:
        t = loc.type
        ll = (
            np.uint32(loc.x) * img_size[0] // map.size[0],
            np.uint32(loc.y) * img_size[1] // map.size[1],
        )
        color = (0,0,0) if t==CUSTOMER else (0,0,255)
        draw_circle(img, ll, color, 5, t[0])
    return img


if __name__ == "__main__":
    # Example with a small map
    map_obj = Map((50, 50), num_cities=3, num_depots=1)
    map_obj.sample_locations(500)
    map_obj.cluster_and_place_depots()
    img = map_drawer(map_obj)
    img.show()
