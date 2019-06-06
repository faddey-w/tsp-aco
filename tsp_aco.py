import random
import os
import itertools
import math


JUMP_PHEROMONE = "naive"
# JUMP_PHEROMONE = 'heuristical'
# JUMP_PHEROMONE = 'nondirected'
USE_JUMP_PHEROMONE_DOWNSCALE = False
DECOUPLE_LONGEST_BY_JUMP = False
DECOUPLE_LONGEST_BY_MOVE = False


class Heuristic:
    def __init__(
        self,
        distances,
        n_salesmans,
        greedness=1,
        herdness=1,
        evaporation=0.1,
        n_probas=None,
        features=(),
    ):
        self.distances = distances
        self.n_salesmans = n_salesmans
        self.n = n = len(distances)
        self.n_probas = n_probas or n
        self.herdness = herdness
        self.evaporation = evaporation
        self.use_jump_balancing = 'JUMP_BALANCING' in features
        mean_d = sum(map(sum, distances)) / (n * n)
        self.move_heuristic = [
            _normalize([(1 / d) ** greedness if d else 0 for d in row])
            for row in distances
        ]
        self.move_utility = [[1 / (n * mean_d)] * n for _ in range(n)]
        self.jump_utility = [[1 / (n * mean_d)] * n for _ in range(n)]
        self.jump_heuristic = [
            [1 / (n - 1) for d in row] for i, row in enumerate(distances)
        ]
        self._reset_update_buffers()
        self._update_evidences()

    def _reset_update_buffers(self):
        n = self.n
        self._move_utility_update = [[0] * n for _ in range(n)]
        self._jump_utility_update = [[0] * n for _ in range(n)]

    def _update_evidences(self):
        n = self.n
        move_utility = self.move_utility
        jump_utility = self.jump_utility
        move_heuristic = self.move_heuristic
        jump_heuristic = self.jump_heuristic
        herdness = self.herdness

        self._move_evidence = [
            [move_heuristic[i][j] * (move_utility[i][j] ** herdness) for j in range(n)]
            for i in range(n)
        ]
        self._jump_evidence = [
            [jump_heuristic[i][j] * (jump_utility[i][j] ** herdness) for j in range(n)]
            for i in range(n)
        ]

    def add_path(self, paths, cost):
        lengths = [path_len(p, self.distances) for p in paths]
        max_length = max(lengths)
        jump_utility_update = self._jump_utility_update
        move_utility_update = self._move_utility_update

        # commit to pheromone update buffer

        upd = (1 / cost) / self.n_probas
        m_upd = upd
        j_upd = upd
        if USE_JUMP_PHEROMONE_DOWNSCALE:
            j_upd /= len(paths)
        for k, pth1 in enumerate(paths):
            if JUMP_PHEROMONE == "heuristical":
                for pth2 in paths[k + 1 :]:
                    for c1, c2 in itertools.product(pth1, pth2):
                        jump_utility_update[c1][c2] += j_upd
                        jump_utility_update[c2][c1] += j_upd
            elif JUMP_PHEROMONE == "naive":
                c1 = pth1[0]
                c2 = paths[k - 1][-1]
                jump_utility_update[c1][c2] += j_upd
                jump_utility_update[c2][c1] += j_upd
            else:
                raise NotImplementedError
            if lengths[k] == max_length and self.n_salesmans > 1:
                if DECOUPLE_LONGEST_BY_JUMP:
                    for c1, c2 in itertools.product(pth1, pth1):
                        if c1 == c2:
                            continue
                        jump_utility_update[c1][c2] += j_upd
                        jump_utility_update[c2][c1] += j_upd
                if DECOUPLE_LONGEST_BY_MOVE:
                    continue
            for i in range(len(pth1)):
                c1, c2 = pth1[i - 1], pth1[i]
                move_utility_update[c2][c1] += m_upd
                move_utility_update[c1][c2] += m_upd
            # for pth2 in proba_paths[k+1:]:
            #     for c1, c2 in itertools.product(pth1, pth2):
            #         jump_utility_update[c1][c2] += j_upd
            #         jump_utility_update[c2][c1] += j_upd
            # jump_utility_update[prev_pth[-1]][pth1[0]] += upd
            # jump_utility_update[pth1[0]][prev_pth[-1]] += upd

        return cost

    def update(self):
        n = self.n
        evaporation = self.evaporation
        move_utility = self.move_utility
        move_utility_update = self._move_utility_update
        jump_utility = self.jump_utility
        jump_utility_update = self._jump_utility_update

        # update pheromone and re-calculate transition probs (M-step)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                move_utility[i][j] = (1 - evaporation) * move_utility[i][
                    j
                ] + evaporation * move_utility_update[i][j]
                min_ju = min(map(min, jump_utility_update))
                jump_utility[i][j] = (1 - evaporation) * jump_utility[i][
                    j
                ] + evaporation * (jump_utility_update[i][j] - min_ju)

        self._update_evidences()
        self._reset_update_buffers()

    def get_move_probs(self, from_node, to_nodes, jumps_left):
        move_utility = self._move_evidence
        jump_utility = self._jump_evidence
        moverow = [move_utility[from_node][i] for i in to_nodes]
        jumprow = [jump_utility[from_node][i] for i in to_nodes]
        if self.use_jump_balancing:
            jump_prob = jumps_left / len(to_nodes)
            move_prob = 1 - jump_prob
        else:
            jump_prob = 1 if jumps_left > 0 else 0
            move_prob = 1 if jumps_left > len(to_nodes) else 0
        return (move_prob, moverow), (jump_prob, jumprow)


def solve_tsp(
    distances,
    heuristic: Heuristic,
    callback=lambda paths, *utilities, **costs: None,
    max_iteration=10,
    n_salesmans=1,
    objective="max",
    _debug=False,
):
    if _debug > 0:
        _debug -= 1
    n = len(distances)
    assert heuristic.n_salesmans < heuristic.n, (
        "if there are no less salesmans than cities, "
        "you do not need to force them to travel"
    )
    assert objective in ("max", "total")
    assert JUMP_PHEROMONE in ("naive", "heuristical")

    best_paths = _make_aco_proba(n, n_salesmans, heuristic, greedy=True)
    best_cost = _get_proba_cost(best_paths, heuristic.distances)
    callback(best_paths, heuristic.move_utility, L=best_cost)

    for _ in range(max_iteration):
        for _ in range(heuristic.n_probas):
            proba_paths = _make_aco_proba(n, n_salesmans, heuristic, greedy=False)
            cost = _get_proba_cost(proba_paths, heuristic.distances)
            if cost < best_cost:
                best_cost = cost
                best_paths = proba_paths
            heuristic.add_path(proba_paths, cost)

        heuristic.update()

        # at each iteration we greedily compute current best solution
        # so callback can see the progress
        greedy_paths = _make_aco_proba(n, n_salesmans, heuristic, greedy=True)
        cost = _get_proba_cost(greedy_paths, heuristic.distances)
        if cost < best_cost:
            best_paths = greedy_paths
            best_cost = cost
        callback(greedy_paths, heuristic.move_utility, L=cost)

    callback(best_paths, heuristic.move_utility, L=best_cost)

    return best_paths


def _get_proba_cost(proba_paths, distances):
    lengths = [path_len(p, distances) for p in proba_paths]
    return max(lengths)


def _format1d(vector):
    return "[" + ", ".join("{:.3f}".format(v) for v in vector) + "]"


def _format2d(matrix):
    return "\n".join(map(_format1d, matrix))


def _make_aco_proba(n, n_salesmans, heuristic, greedy):
    not_visited = list(range(n))
    paths = [[0]]
    not_visited.remove(paths[0][0])
    jumps_left = n_salesmans - 1
    for _ in range(n - 1):
        pth = paths[-1]
        c = pth[-1]
        (move_prob, moverow), (jump_prob, jumprow) = heuristic.get_move_probs(
            c, not_visited, jumps_left
        )
        if greedy:
            jnc_p = _argmax(jumprow)
            mnc_p = _argmax(moverow)
            jump = jump_prob * jumprow[jnc_p] > move_prob * moverow[mnc_p]
        else:
            jnc_p = _weighted_choice(jumprow)
            mnc_p = _weighted_choice(moverow)

            jp = jump_prob * jumprow[jnc_p]
            mp = move_prob * moverow[mnc_p]
            jump = _weighted_choice([mp, jp])
        nc_p = jnc_p if jump else mnc_p
        nc = not_visited.pop(nc_p)
        if jump:
            paths.append([nc])
            jumps_left -= 1
        else:
            pth.append(nc)
    assert len(paths) == n_salesmans
    return paths


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
    return [e / expsum for e in exps]


def _normalize(vector, norm=1.0):
    factor = norm / sum(vector)
    return [v * factor for v in vector]


def _shift_to_positive(matrix):
    minval = min(map(min, matrix))
    return [[x - minval for x in row] for row in matrix]


def path_len(path, distances):
    d = 0
    for i in range(len(path)):
        d += distances[path[i - 1]][path[i]]
    return d


class GenerateVisualSvg:

    _width = 500
    _text_height = 50

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
                if fname.endswith(".svg"):
                    fpath = os.path.join(savedir, fname)
                    os.remove(fpath)
        else:
            os.makedirs(savedir)

    def __call__(self, paths, *utilities, **costs):
        width = self._width + 10
        contents = """
        <?xml version="1.0" encoding="utf-8" ?>
        <svg width="{}" height="{}" version="1.1"
             baseProfile="full" xmlns="http://www.w3.org/2000/svg">
        """.lstrip().format(
            len(utilities) * width, width + self._text_height
        )
        contents += "<style>.txt {font: bold 26px sans-serif;}</style>"

        costs_text = "; ".join(
            "{}={:2f}".format(key, val) for key, val in costs.items()
        )
        paths_text = repr(paths)

        contents += '<text x="25" y="25" class="txt">{} paths={}</text>'.format(
            costs_text, paths_text
        )
        for i, utility in enumerate(utilities):
            contents += self._render_graph(utility, paths, i * width)
            paths = []
        contents += "</svg>"

        with open(os.path.join(self.savedir, "{}.svg".format(self.i)), "w") as f:
            f.write(contents)
        self.i += 1

    def _render_graph(self, edge_weights, highlighted_paths, x_offset):
        contents = ""
        max_utility = max(map(max, edge_weights))
        pathpairs = set(
            itertools.chain(*(zip(p[:-1], p[1:]) for p in highlighted_paths))
        )
        pathpairs.update((p[-1], p[0]) for p in highlighted_paths)
        w = self._width
        th = self._text_height

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
                    x_offset + w * (self.coordinates[i][0] - self.xoffs) / self.xrange,
                    th + w * (self.coordinates[i][1] - self.yoffs) / self.yrange,
                    x_offset + w * (self.coordinates[j][0] - self.xoffs) / self.xrange,
                    th + w * (self.coordinates[j][1] - self.yoffs) / self.yrange,
                    max(minw, 3 * edge_weights[i][j] / max_utility),
                    ",".join(map(str, color)),
                )
                contents += line
        return contents
