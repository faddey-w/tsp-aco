import random
import os
import itertools
import math


def solve_tsp(distances, callback=lambda utility, path: None,
              n_probas=None,
              herdness=1,
              greedness=1,
              evaporation=0.1,
              max_iteration=10,
              n_salesmans=1,
              objective='max',
              _debug=True):
    n = len(distances)
    n_probas = n_probas or n
    assert n_salesmans < n, "if there are no less salesmans than cities, " \
                            "you do not need to force them to travel"
    assert objective in ('max', 'total')

    mean_d = sum(map(sum, distances)) / (n*n)
    min_d = min(min(filter(bool, row)) for row in distances)

    # Walk properties are responsible for building optimal paths between nodes.
    # It is from classic ACO algorithm

    # walk utility is amount of ant's pheromone on the path between nodes
    walk_utility = [[1 / (mean_d*n)] * n for _ in range(n)]
    # walk_utility = [[1 / (min_d*n)] * n for _ in range(n)]
    dissimilarity = [
        _softmax(row)
        for row in distances
    ]
    cluster_utility = [
        [1/n for _ in range(n)]
        for _ in range(n)
    ]
    walk_heuristic = [
        [
            (1 / d) ** greedness if d else 0
            for d in row
        ]
        for row in distances
    ]
    if _debug:
        print('initial walk_utility =', 1 / (mean_d*n))

    # assignment properties are responsible for multi-TSP extension
    # They manage clustering of initial graph into groups

    # assignment_utility = [
    #     # [sum(row) / n for row in walk_heuristic]
    #     [1/n for _ in range(n)]
    #     for _ in range(n_salesmans)
    # ]

    # evidence is unscaled probabilities of each transition on current round.
    # initial evidence is proportional to heuristic, so don't do anything here
    evidence = walk_heuristic
    clustership = [
        [
            (1 / ds)
            for ds in ds_row
        ]
        for ds_row in dissimilarity
    ]

    best_paths = _make_aco_proba(
            n, n_salesmans,
            evidence, clustership,
            greedy=True)

    callback(walk_utility, cluster_utility, best_paths)

    for _ in range(max_iteration):
        # there is no early-stop criteria actually

        # assignment_utility_sums = list(map(sum, assignment_utility))

        # accumulate new pheromone in separate buffer
        # to keep transition probabilities same during one round of probas
        walk_utility_update = [[0] * n for _ in range(n)]
        cluster_utility_update = [[0] * n for _ in range(n)]

        # 1) make randomized probas several times (E-step)
        # 2) calculate the objective and pheromone update for each (pre-M-step)
        for _ in range(n_probas):
            # do the proba
            proba_paths = _make_aco_proba(
                n, n_salesmans,
                evidence, clustership,
                greedy=False)

            # calculate the objective
            lengths = []
            for pth in proba_paths:
                length = 0
                for i in range(1, len(pth)):
                    length += distances[i-1][i]
                if length == 0:
                    length = 1
                lengths.append(length)
            max_length = max(lengths)
            total_length = sum(lengths)
            mean_length = total_length / len(lengths)

            if objective == 'max':
                cost = max_length
            elif objective == 'total':
                cost = total_length
            else:
                raise NotImplementedError(objective)

            # commit to pheromone update buffer

            for s, pth in enumerate(proba_paths):
                l = lengths[s]
                walk_upd = (1 / l) / n_probas
                cluster_upd = 1 / max(1, l)
                # cluster_upd = 1 / (lengths[s]-mean_length) / n_probas
                # if lengths[s] == max_length:
                #     cluster_upd = -lengths[s]
                # else:
                #     cluster_upd = 1 / lengths[s]
                for i in range(len(pth)):
                    c1, c2 = pth[i-1], pth[i]
                    walk_utility_update[c2][c1] += walk_upd
                    walk_utility_update[c1][c2] += walk_upd
                    # if c1 != c2:
                    cluster_utility_update[c2][c1] += cluster_upd
                    cluster_utility_update[c1][c2] += cluster_upd

            if _debug:
                print('proba_paths =', proba_paths)
        if _debug:
            print('walk_utility_update')
            print(_format2d(walk_utility_update))
            print('cluster_utility_update')
            print(_format2d(cluster_utility_update))

        # update pheromone and re-calculate transition probs (M-step)
        for i in range(n):
            cl_upd = _softmax(cluster_utility_update[i])
            for j in range(n):
                if i == j:
                    continue
                walk_utility[i][j] = (
                    (1-evaporation) * walk_utility[i][j]
                    + evaporation * walk_utility_update[i][j]
                )
                # cluster_utility[i][j] = (
                #     (1 - evaporation) * cluster_utility[i][j]
                #     + evaporation * cluster_utility_update[i][j]
                # )
                cl_ev = 0.1
                cluster_utility[i][j] = (
                    (1 - cl_ev) * cluster_utility[i][j]
                    + cl_ev * cl_upd[j]
                )
            # min_cu = min(cluster_utility[i])
            # sum_cu = sum(cluster_utility[i])
            # normer =
            # cluster_utility[i] = _softmax(cluster_utility[i])
        if _debug:
            print('walk_utility')
            print(_format2d(walk_utility))
            print('cluster_utility')
            print(_format2d(cluster_utility))

        evidence = [
            [
                walk_heuristic[i][j] * (walk_utility[i][j] ** herdness)
                for j in range(n)
            ]
            for i in range(n)
        ]
        clustership = [
            [
                (cluster_utility[i][j] ** 1) / (dissimilarity[i][j] ** greedness)
                # (cluster_utility[i][j] ** herdness) / (dissimilarity[i][j] ** greedness)
                for j in range(n)
            ]
            for i in range(n)
        ]

        # at each iteration we greedily compute current best solution
        # so callback can see the progress
        best_paths = _make_aco_proba(
            n, n_salesmans,
            evidence, clustership,
            greedy=True)
        callback(walk_utility, cluster_utility, best_paths)

    return best_paths


def _format1d(vector):
    return '[' + ', '.join('{:.3f}'.format(v) for v in vector) + ']'


def _format2d(matrix):
    return '\n'.join(map(_format1d, matrix))


def _make_aco_proba(n, n_salesmans, evidence, clustership, greedy):
    # initially assign start cities to salesmans
    # according to assignment utility
    assignment = [0] * n_salesmans
    clshsums = list(map(sum, clustership))
    _unassigned_cities = list(range(1, n))
    for s in range(1, n_salesmans):
        if greedy:
            c_p = _argmax([
                clustership[assignment[_s]][c]
                for _s in range(s)
                for c in _unassigned_cities
            ]) % s
        else:
            pre_s_p = _weighted_choice([
                clshsums[assignment[_s]]
                for _s in range(s)
            ])
            c_p = _weighted_choice([
                clustership[assignment[pre_s_p]][c]
                for c in _unassigned_cities
            ])
        c = _unassigned_cities.pop(c_p)

        assignment[s] = c
    curr_city = assignment[:]

    not_visited = _unassigned_cities
    proba_paths = [[] for _ in range(n_salesmans)]

    have_many_salesmans = n_salesmans > 1
    active_salesmans = list(range(n_salesmans))
    evirows = []
    for s, pth in enumerate(proba_paths):
        asmt = assignment[s]
        if have_many_salesmans:
            # evirow = [evidence[asmt][c] * clustership[asmt][c] for c in not_visited]
            evirow = [evidence[asmt][c] for c in not_visited]
            evirow.append(evidence[asmt][asmt])
        else:
            evirow = [evidence[asmt][c] for c in not_visited]
        evirows.append(evirow)

    if greedy:
        evisums = None
    else:
        evisums = list(map(sum, evirows))

    for _ in range(n-1):

        if greedy:
            s_p, nc_p = _argmax2d(evirows)
        else:
            s_p = _weighted_choice(evisums)
            nc_p = _weighted_choice(evirows[s_p])

        s = active_salesmans[s_p]
        pth = proba_paths[s]
        if nc_p == len(not_visited):
            assert have_many_salesmans
            c = assignment[s]

            del active_salesmans[s_p]
            del evirows[s_p]
            if evisums:
                del evisums[s_p]

            if len(active_salesmans) == 1:
                e = evirows[0].pop(-1)
                if evisums:
                    evisums[0] -= e
                have_many_salesmans = False

        else:
            c = not_visited[nc_p]
            del not_visited[nc_p]

            for _s_p in range(len(active_salesmans)):
                if s_p != _s_p:
                    if evisums:
                        evisums[_s_p] -= evirows[_s_p][nc_p]
                    del evirows[_s_p][nc_p]
            if have_many_salesmans:
                # evirows[s_p] = [evidence[c][nc] * clustership[c][nc] for nc in not_visited]
                evirows[s_p] = [evidence[c][nc] for nc in not_visited]
                evirows[s_p].append(evidence[c][assignment[s]])
            else:
                evirows[s_p] = [evidence[c][nc] for nc in not_visited]
            if evisums:
                evisums[s_p] = sum(evirows[s_p])

        curr_city[s] = c
        pth.append(c)

    assert not have_many_salesmans
    assert len(active_salesmans) == 1
    assert not_visited == []
    s = active_salesmans[0]
    assert not any(assignment[s] in pth for pth in proba_paths)
    proba_paths[s].append(assignment[s])

    return proba_paths


def _weighted_choice(weights):
    wsum = sum(weights)
    threshold = random.random() * wsum
    wacc = 0
    for i, w in enumerate(weights):
        wacc += w
        if wacc >= threshold:
            return i
    return 0


def _argmax2d(matrix):
    nrows = len(matrix)
    ncols = len(matrix[0])
    rows_max_i = [max(range(ncols), key=lambda c: row[c]) for row in matrix]
    max_r = max(range(nrows), key=lambda r: matrix[r][rows_max_i[r]])
    max_c = rows_max_i[max_r]
    return max_r, max_c


def _argmax(weights):
    return max(range(len(weights)), key=lambda i: weights[i])


def _softmax(vector):
    exps = [math.exp(v) for v in vector]
    expsum = sum(exps)
    return [e/expsum for e in exps]


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
            for fname in os.listdir(savedir):
                if fname.endswith('.svg'):
                    fpath = os.path.join(savedir, fname)
                    os.remove(fpath)
        else:
            os.makedirs(savedir)

    def __call__(self, walk_utility, cluster_utility, paths):
        contents = """
        <?xml version="1.0" encoding="utf-8" ?>
        <svg width="1000" height="500" version="1.1"
             baseProfile="full" xmlns="http://www.w3.org/2000/svg">
        """.lstrip()
        contents += self._render_graph(walk_utility, paths, 0)
        contents += self._render_graph(cluster_utility, [], 500)
        contents += "</svg>"

        with open(os.path.join(self.savedir, '{}.svg'.format(self.i)), 'w') as f:
            f.write(contents)
        self.i += 1

    def _render_graph(self, edge_weights, highlighted_paths, x_offset):
        contents = ''
        max_utility = max(map(max, edge_weights))
        pathpairs = set(itertools.chain(*(
            zip(p[:-1], p[1:])
            for p in highlighted_paths
        )))
        pathpairs.update((p[-1], p[0]) for p in highlighted_paths)

        n = len(edge_weights)
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
                    x_offset + 500 * (self.coordinates[i][0] - self.xoffs) / self.xrange,
                    500 * (self.coordinates[i][1] - self.yoffs) / self.yrange,
                    x_offset + 500 * (self.coordinates[j][0] - self.xoffs) / self.xrange,
                    500 * (self.coordinates[j][1] - self.yoffs) / self.yrange,
                    max(minw, 3 * edge_weights[i][j] / max_utility),
                    ','.join(map(str, color)),
                )
                contents += line
        return contents
