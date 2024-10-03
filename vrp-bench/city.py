from typing import Tuple, List
import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass

from constants import CUSTOMER, DEPOT


@dataclass
class Location:
    x: float
    y: float
    type: str = CUSTOMER

    def distance(self, other: "Location") -> int:
        return np.linalg.norm(
            np.array([self.x, self.y]) - np.array([other.x, other.y])
        ).astype(int)


class City:
    def __init__(self, center: Tuple[float, float], spread: float):
        self.center = np.array(center)
        self.spread = spread

    def sample(self, map_size: Tuple[int, int]) -> Location:
        sample = np.random.normal(self.center, self.spread)
        loc = np.clip(sample, [0, 0], map_size).astype(int)
        return Location(loc[0], loc[1])

    def batch_sample(self, map_size: Tuple[int, int], n: int) -> List[Location]:
        samples = np.random.normal(self.center, self.spread, size=(n, 2))
        locations = np.clip(samples, [0, 0], map_size).astype(int)
        return [Location(loc[0], loc[1]) for loc in locations]

    def __repr__(self):
        return f"City(center={self.center}, spread={self.spread})"


class Map:
    spread_range = (5, 50)

    def __init__(self, size: Tuple[int, int], num_cities: int, num_depots: int):
        self.size = size

        self.cities = [
            City(
                (np.random.randint(size[0]), np.random.randint(size[1])),
                np.random.randint(*self.spread_range),
            )
            for _ in range(num_cities)
        ]
        self.depots = self.generate_depots(num_depots)
        self.locations = []

    def generate_depots(self, num_depots):
        depots = []
        for _ in range(num_depots):
            assigned_city = np.random.choice(self.cities)
            x = np.random.normal(assigned_city.center[0], 2 * assigned_city.spread)
            y = np.random.normal(assigned_city.center[1], 2 * assigned_city.spread)
            depots.append(Location(x, y, DEPOT))
        return depots

    def sample_locations(self, num_locations: int) -> List[Location]:
        locations = []
        loc_per_city = num_locations // len(self.cities)
        for city in self.cities:
            locations.extend(city.batch_sample(self.size, loc_per_city))
        locations.extend(self.depots)
        self.locations = self.depots + locations
        return locations

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
    # for city in map.cities:
    #     center = city.center
    #     # scale
    #     center = (
    #         center[0] * img_size[0] // map.size[0],
    #         center[1] * img_size[1] // map.size[1],
    #     )
    #     draw_circle(img, center, (255, 0, 0), 10, "CT")
    for depot in map.depots:
        depot = (
            depot.x * img_size[0] // map.size[0],
            depot.y * img_size[1] // map.size[1],
        )
        draw_circle(img, depot, (0, 0, 255), 15, "D")

    for loc in map.locations:
        t = loc.type
        loc = (
            loc.x * img_size[0] // map.size[0],
            loc.y * img_size[1] // map.size[1],
        )
        draw_circle(img, loc, (0, 0, 0), 5, t[0])
    return img


if __name__ == "__main__":
    city = City((50, 50), 10)
    print(city.sample((100, 100)))
    map = Map((100, 100), 1, 1)
    print(map.sample_locations(500))
    img = map_drawer(map)
    img.show()
