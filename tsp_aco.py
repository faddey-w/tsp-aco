import random
import os
import pprint
import shutil


def solve_tsp(distances, callback=lambda utility, path: None,
              n_probes=None,
              utility_scale=1,
              herdness=1,
              greedness=1,
              evaporation=0.1,
              max_iteration=10,
              _debug=False):
    n = len(distances)
    n_probes = n_probes or n
    utility = [[utility_scale/n] * n for _ in range(n)]
    heuristic = [
        [
            (1 / d) ** greedness if d else 0
            for d in row
        ]
        for row in distances
    ]
    best_path = list(range(n))

    callback(utility, best_path)

    for _ in range(max_iteration):
        utility_update = [[0] * n for _ in range(n)]
        evidence = [
            [
                heuristic[i][j] * (utility[i][j] ** herdness)
                for j in range(n)
            ]
            for i in range(n)
        ]

        for _ in range(n_probes):
            not_visited = set(range(n))
            last = 0
            probe_path = [last]
            for _ in range(n-1):
                not_visited.remove(last)
                evirow = evidence[last]
                evisum = sum(evirow[i] for i in not_visited)
                cumevi = 0
                threshold = evisum * random.random()

                next_ = None
                for i in not_visited:
                    cumevi += evirow[i]
                    if cumevi >= threshold:
                        next_ = i
                        break

                upd = utility_scale * heuristic[last][next_]
                utility_update[last][next_] += upd
                utility_update[next_][last] += upd
                probe_path.append(next_)
                last = next_
            if _debug:
                print(probe_path)
        if _debug:
            pprint.pprint(utility_update)

        for i in range(n):
            for j in range(n):
                utility[i][j] = (
                    (1-evaporation) * utility[i][j]
                    + utility_update[i][j]
                    + utility_update[j][i]
                )

        best_path = []
        not_visited = set(range(n))
        current = 0
        while not_visited:
            next_ = max(not_visited, key=evidence[current].__getitem__)
            best_path.append(next_)
            current = next_
            not_visited.remove(next_)
        callback(utility, best_path)

    return best_path


class GenerateVisualSvg:

    def __init__(self, savedir, coordinates):
        self.savedir = savedir
        self.coordinates = coordinates
        minx = min(x for x, y in coordinates)
        maxx = max(x for x, y in coordinates)
        miny = min(y for x, y in coordinates)
        maxy = max(y for x, y in coordinates)
        self.xoffs = minx
        self.xrange = maxx - minx
        self.yoffs = miny
        self.yrange = maxy - miny
        self.i = 0

        if os.path.exists(savedir):
            shutil.rmtree(savedir)
        os.makedirs(savedir)

    def __call__(self, utility, path):
        contents = """
        <?xml version="1.0" encoding="utf-8" ?>
        <svg width="500" height="500" version="1.1"
             baseProfile="full" xmlns="http://www.w3.org/2000/svg">
        """.lstrip()
        max_utility = max(map(max, utility))
        pathpairs = set(zip(path[:-1], path[1:]))
        pathpairs.add((path[-1], path[0]))

        n = len(path)
        for i in range(n):
            for j in range(i):
                if (i, j) in pathpairs or (j, i) in pathpairs:
                    color = 255, 0, 0
                    minw = 1
                else:
                    color = 0, 0, 0
                    minw = 0.01

                line = """
                <line x1="{}" y1="{}" x2="{}" y2="{}" stroke-width="{}" stroke="rgb({})"/>
                """.format(
                    500 * (self.coordinates[i][0] - self.xoffs) / self.xrange,
                    500 * (self.coordinates[i][1] - self.yoffs) / self.yrange,
                    500 * (self.coordinates[j][0] - self.xoffs) / self.xrange,
                    500 * (self.coordinates[j][1] - self.yoffs) / self.yrange,
                    max(minw, 3 * utility[i][j] / max_utility),
                    ','.join(map(str, color)),
                )
                contents += line
        contents += "</svg>"

        with open(os.path.join(self.savedir, '{}.svg'.format(self.i)), 'w') as f:
            f.write(contents)
        self.i += 1
