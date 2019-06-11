import random
import math
import argparse
import os
import csv
import time
import string


def generate_graph(n, form, scale, center=None):
    if center is None:
        center = (0, 0)
    if form == "gauss":

        def get_node():
            return random.gauss(0, 1), random.gauss(0, 1)

    elif form == "uniform":

        def get_node():
            return random.random(), random.random()

    elif form == "circle":

        def get_node():
            r = 1 + 0.1 * random.random()
            a = math.pi * 2 * random.random()
            return r * math.cos(a), r * math.sin(a)

    else:
        raise ValueError(f"form = {form!r} is invalid")
    cx, cy = center
    g = []
    for _ in range(n):
        x, y = get_node()
        g.append((cx + scale * x, cy + scale * y))
    return g


def readfile(path, country=None):
    with open(path, encoding="ISO-8859-1") as f:
        if country is None:
            iterlines = f
        else:
            iterlines = (l for i, l in enumerate(f) if l.startswith(country) or i == 0)
        rows = list(csv.reader(iterlines))
    heading = rows.pop(0)
    return heading, rows


def parse_graph_from_file(path, country=None, top_n=None):
    heading, rows = readfile(path, country)
    # heading = Country, City, AccentCity, Region, Population, Lat, Lon
    #           0        1     2           3       4           5    6
    if top_n is None:
        cities = rows
    else:
        populations = [int(r[4] or 0) for r in rows]
        top_pop_idx = sorted(
            range(len(rows)), key=populations.__getitem__, reverse=True
        )
        cities = [rows[i] for i in top_pop_idx[:top_n]]

    lat = [float(r[5]) for r in cities]
    lon = [float(r[6]) for r in cities]
    coordinates = list(zip(lat, lon))
    city_names = [c[1] for c in cities]

    distmat = [
        [
            ((lat[i1] - lat[i2]) ** 2 + (lon[i1] - lon[i2]) ** 2) ** 0.5
            for i2 in range(len(cities))
        ]
        for i1 in range(len(cities))
    ]
    return city_names, coordinates, distmat


def name_generator():
    alphabet = string.ascii_uppercase
    n = len(alphabet)
    i = 0
    while True:
        letters_reversed = []
        x = i
        while True:
            r = x % n
            letters_reversed.append(alphabet[r])
            x //= n
            if x == 0:
                break
        yield "".join(letters_reversed[::-1])
        i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params", nargs="+")
    parser.add_argument("--seed", "-s", type=int)
    opts = parser.parse_args()
    params_encoded = opts.params
    if opts.seed is None:
        opts.seed = hash(time.time())
    random.seed(opts.seed)
    assert len(params_encoded) > 0
    graph = []
    filename = f"{opts.seed}-{':'.join(params_encoded)}.csv"
    for params_str in params_encoded:
        parts = params_str.split(",")
        assert len(parts) in (3, 5), parts
        n, form, scale = parts[:3]
        n = int(n)
        scale = float(scale)
        if len(parts) == 5:
            center = tuple(map(float, parts[3:5]))
        else:
            center = None
        graph.extend(generate_graph(n, form, scale, center))

    os.makedirs("graphs", exist_ok=True)
    with open(f"graphs/{filename}", "w") as f:
        writer = csv.writer(f)
        writer.writerow("Country City AccentCity Region Population Lat Lon".split())
        for name, (x, y) in zip(name_generator(), graph):
            writer.writerow(["x", name, "", "", "1", str(x), str(y)])


if __name__ == "__main__":
    main()
