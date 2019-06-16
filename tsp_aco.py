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
    def __init__(self, distances, n_salesmans, n_probas=None):
        self.distances = distances
        self.n_salesmans = n_salesmans
        self.n = len(distances)
        self.n_probas = n_probas or self.n

    def get_utilities(self):
        return ()

    def add_solution(self, paths, cost):
        pass

    def update(self):
        pass

    def __call__(self, path, to_nodes, n_paths):
        n = len(to_nodes)
        return [1 / (2 * n)] * n, [1 / (2 * n)] * n


class HeuristicV1(Heuristic):
    def __init__(
        self,
        distances,
        n_salesmans,
        n_probas=None,
        greedness=1,
        herdness=1,
        evaporation=0.1,
        features=(),
    ):
        super(HeuristicV1, self).__init__(distances, n_salesmans, n_probas)
        n = self.n
        self.herdness = herdness
        self.evaporation = evaporation
        self.use_jump_balancing = "JUMP_BALANCING" in features
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

    def get_utilities(self):
        return self.move_utility, self.jump_utility

    def add_solution(self, paths, cost):
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

    def __call__(self, path, to_nodes, n_paths):
        from_node = path[-1]
        jumps_left = self.n_salesmans - n_paths
        move_utility = self._move_evidence
        jump_utility = self._jump_evidence
        if self.use_jump_balancing:
            jump_prob = jumps_left / len(to_nodes)
            move_prob = 1 - jump_prob
        else:
            if jumps_left >= len(to_nodes):
                jump_prob = 1
                move_prob = 0
            elif jumps_left > 0:
                jump_prob = move_prob = 1
            else:
                jump_prob = 0
                move_prob = 1
        moverow = [move_utility[from_node][i] for i in to_nodes]
        jumprow = [jump_utility[from_node][i] for i in to_nodes]
        total = move_prob * sum(moverow) + jump_prob * sum(jumprow)
        moveprobs = [move_prob * m / total for m in moverow]
        jumpprobs = [jump_prob * j / total for j in jumprow]
        assert abs(sum(moveprobs) + sum(jumpprobs) - 1) < 1e-5
        return moveprobs, jumpprobs


class HeuristicV2(Heuristic):
    def __init__(
        self,
        distances,
        n_salesmans,
        n_probas=None,
        greedness=1,
        herdness=1,
        evaporation=0.1,
    ):
        super(HeuristicV2, self).__init__(distances, n_salesmans, n_probas)
        n = self.n
        self.herdness = herdness
        self.evaporation = evaporation
        mean_d = sum(map(sum, distances)) / (n * n)
        self.move_heuristic = [
            [(1 / d) ** greedness if d else 1 for d in row] for row in distances
        ]
        self._move_utility = [[1 / (n * mean_d)] * n for _ in range(n)]
        self._clusterity = [[1 / (n * mean_d)] * n for _ in range(n)]
        self._reset_update_buffers()

    def _reset_update_buffers(self):
        n = self.n
        self._move_utility_update = [[0] * n for _ in range(n)]
        self._clusterity_update = [[0] * n for _ in range(n)]

    def get_utilities(self):
        return self._move_utility, self._clusterity

    def add_solution(self, paths, cost):
        lengths = [path_len(p, self.distances) for p in paths]
        clusterity_update = self._clusterity_update
        move_utility_update = self._move_utility_update

        # commit to pheromone update buffer

        for pth_cost, pth in zip(lengths, paths):
            upd = (1 / (cost + pth_cost)) / self.n_probas
            m_upd = upd
            c_upd = upd
            for i in range(len(pth)):
                c1, c2 = pth[i - 1], pth[i]
                move_utility_update[c2][c1] += m_upd
                move_utility_update[c1][c2] += m_upd
            for i, c1 in enumerate(pth):
                for j in range(i + 1, len(pth)):
                    c2 = pth[j]
                    clusterity_update[c1][c2] += c_upd
                    clusterity_update[c2][c1] += c_upd

        return cost

    def update(self):
        n = self.n
        evaporation = self.evaporation
        move_utility = self._move_utility
        move_utility_update = self._move_utility_update
        clusterity = self._clusterity
        clusterity_update = self._clusterity_update

        leave = 1 - evaporation
        add = 1
        for i in range(n):
            for j in range(n):
                move_utility[i][j] = (
                    leave * move_utility[i][j] + add * move_utility_update[i][j]
                )
                clusterity[i][j] = (
                    leave * clusterity[i][j] + add * clusterity_update[i][j]
                )

        self._reset_update_buffers()

    def __call__(self, path, to_nodes, n_paths):
        jump_prob = (self.n_salesmans - n_paths) / len(to_nodes)
        move_prob = 1 - jump_prob
        from_node = path[-1]
        start_node = path[0]
        u = self._move_utility[from_node]
        a = self.herdness
        h = self.move_heuristic[from_node]
        c = self._clusterity[from_node]
        move_values = [move_prob * (u[i] ** a) * h[i] * c[i] for i in to_nodes]
        jump_value = jump_prob * (u[start_node] ** a) * h[start_node] * c[start_node]
        total = sum(move_values) + sum(jump_value)
        moveprobs = [m / total for m in move_values]
        jumpprobs = [jump_value / len(to_nodes) / total] * len(to_nodes)
        assert abs(sum(moveprobs) + sum(jumpprobs) - 1) <= 0.00001
        return moveprobs, jumpprobs


def solve_tsp(
    distances, heuristic: Heuristic, callback=None, max_iteration=10, _debug=False
):
    if callback is None:
        callback = lambda paths, *utilities, **costs: None
    if _debug > 0:
        _debug -= 1
    n = len(distances)
    assert heuristic.n_salesmans < heuristic.n, (
        "if there are no less salesmans than cities, "
        "you do not need to force them to travel"
    )

    best_paths = _make_aco_proba(n, heuristic, greedy=True)
    assert len(best_paths) == heuristic.n_salesmans
    best_cost = _get_proba_cost(best_paths, heuristic.distances)
    callback(best_paths, *heuristic.get_utilities(), L=best_cost)

    for _ in range(max_iteration):
        for _ in range(heuristic.n_probas):
            proba_paths = _make_aco_proba(n, heuristic, greedy=False)
            cost = _get_proba_cost(proba_paths, heuristic.distances)
            if cost < best_cost:
                best_cost = cost
                best_paths = proba_paths
            heuristic.add_solution(proba_paths, cost)

        heuristic.update()

        # at each iteration we greedily compute current best solution
        # so callback can see the progress
        greedy_paths = _make_aco_proba(n, heuristic, greedy=True)
        assert len(greedy_paths) == heuristic.n_salesmans
        cost = _get_proba_cost(greedy_paths, heuristic.distances)
        if cost < best_cost:
            best_paths = greedy_paths
            best_cost = cost
        callback(greedy_paths, *heuristic.get_utilities(), L=cost)

    callback(best_paths, *heuristic.get_utilities(), L=best_cost, is_last=True)

    return best_cost, best_paths


def _get_proba_cost(proba_paths, distances):
    lengths = [path_len(p, distances) for p in proba_paths]
    return max(lengths)


def _make_aco_proba(n, heuristic, greedy):
    not_visited = list(range(n))
    paths = [[0]]
    not_visited.remove(paths[0][0])
    for _ in range(n - 1):
        pth = paths[-1]
        moveprobs, jumpprobs = heuristic(pth, not_visited, len(paths))
        if greedy:
            jnc_p = _argmax(jumpprobs)
            mnc_p = _argmax(moveprobs)
            jump = jumpprobs[jnc_p] > moveprobs[mnc_p]
            nc_p = jnc_p if jump else mnc_p
        else:
            nc_p = _weighted_choice(moveprobs + jumpprobs, wsum=1)
            if nc_p < len(not_visited):
                jump = False
            else:
                jump = True
                nc_p -= len(not_visited)
        nc = not_visited.pop(nc_p)
        if jump:
            paths.append([nc])
        else:
            pth.append(nc)
    return paths


def _weighted_choice(weights, wsum=None):
    if wsum is None:
        wsum = sum(weights)
    threshold = random.random() * wsum
    wacc = 0
    for i, w in enumerate(weights):
        wacc += w
        if wacc >= threshold:
            return i
    # return 0
    raise RuntimeError(weights)


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
        self.i = 1

        if os.path.exists(savedir):
            for fname in os.listdir(savedir):
                if fname.endswith(".svg"):
                    fpath = os.path.join(savedir, fname)
                    os.remove(fpath)
        else:
            os.makedirs(savedir)

    def __call__(self, paths, *utilities, is_last=False, **costs):
        width = self._width + 10
        contents = """
        <?xml version="1.0" encoding="utf-8" ?>
        <svg width="{}" height="{}" version="1.1"
             baseProfile="full" xmlns="http://www.w3.org/2000/svg">
        """.lstrip().format(
            len(utilities) * width, width + self._text_height
        )
        contents += """<style>
            .txt {font: bold 15px sans-serif;} 
            .lbl {font: bold 9px sans-serif; text-anchor: middle;}
        </style>"""

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
        if is_last:
            with open(os.path.join(self.savedir, "0.svg"), "w") as f:
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

        black_lines = []
        red_lines = []
        n = len(edge_weights)
        for i in range(n):
            for j in range(i):
                if (i, j) in pathpairs or (j, i) in pathpairs:
                    color = 255, 0, 0
                    minw = 1
                    add_w = 1
                    line_list = red_lines
                else:
                    color = 0, 0, 0
                    minw = 0.01
                    add_w = 0
                    line_list = black_lines

                line = """
                <line x1="{}" y1="{}" x2="{}" y2="{}" stroke-width="{}" stroke="rgb({})"/>
                """.format(
                    x_offset + w * (self.coordinates[i][0] - self.xoffs) / self.xrange,
                    th + w * (self.coordinates[i][1] - self.yoffs) / self.yrange,
                    x_offset + w * (self.coordinates[j][0] - self.xoffs) / self.xrange,
                    th + w * (self.coordinates[j][1] - self.yoffs) / self.yrange,
                    max(minw, add_w + 3 * edge_weights[i][j] / max_utility),
                    ",".join(map(str, color)),
                )
                line_list.append(line)
        contents += "".join(black_lines)
        contents += "".join(red_lines)

        for i in range(n):
            x = x_offset + w * (self.coordinates[i][0] - self.xoffs) / self.xrange
            y = th + w * (self.coordinates[i][1] - self.yoffs) / self.yrange
            point = f"""
            <circle cx="{x}" cy="{y}" r="5" fill="rgb(0, 255, 0)"/>
            <text x="{x}" y="{y+3.5}" class="lbl">{i}</text>
            """
            contents += point

        return contents
