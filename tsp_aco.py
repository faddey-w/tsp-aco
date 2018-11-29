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
              _debug=False):
    if _debug > 0:
        _debug -= 1
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
    move_utility = [[1 / (n*mean_d)] * n for _ in range(n)]
    jump_utility = [[1 / (n*mean_d)] * n for _ in range(n)]
    move_heuristic = [
        _softmax([
            (1 / d) ** greedness if d else 0
            for d in row
        ])
        for row in distances
    ]
    jump_heuristic = [
        _softmax([
            1
            # d ** greedness
            for d in row
        ])
        for row in distances
    ]
    if _debug:
        print('initial move_utility =', move_utility[0][0])
        print('initial jump_utility =', jump_utility[0][0])

    # evidences are unscaled probabilities of each transition on current round.
    # initial evidence is proportional to heuristic, so don't do anything here
    m_evidence = move_heuristic
    j_evidence = jump_heuristic

    best_paths = _make_aco_proba(
            n, n_salesmans,
            m_evidence,
            j_evidence,
            greedy=True)
    if _debug:
        print('initial m_evidence')
        print(_format2d(m_evidence))
        print('initial j_evidence')
        print(_format2d(j_evidence))

    callback(best_paths, move_utility, j_evidence)

    for _ in range(max_iteration):
        # there is no early-stop criteria actually

        # accumulate new pheromone in separate buffer
        # to keep transition probabilities same during one round of probas
        move_utility_update = [[0] * n for _ in range(n)]
        jump_utility_update = [[0] * n for _ in range(n)]

        # 1) make randomized probas several times (E-step)
        # 2) calculate the objective and pheromone update for each (pre-M-step)
        for _ in range(n_probas):
            # do the proba
            proba_paths = _make_aco_proba(
                n, n_salesmans,
                m_evidence,
                j_evidence,
                greedy=False)

            # calculate the objective
            lengths = [_path_len(p, distances) for p in proba_paths]
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

            upd = (1 / cost) / n_probas
            m_upd = upd
            j_upd = upd / len(proba_paths)
            for k, pth1 in enumerate(proba_paths):
                if lengths[k] == max_length:
                    continue
                for i in range(len(pth1)):
                    c1, c2 = pth1[i-1], pth1[i]
                    move_utility_update[c2][c1] += m_upd
                    move_utility_update[c1][c2] += m_upd
                for pth2 in proba_paths[k+1:]:
                    for c1, c2 in itertools.product(pth1, pth2):
                        jump_utility_update[c1][c2] += j_upd
                        jump_utility_update[c2][c1] += j_upd
                # jump_utility_update[prev_pth[-1]][pth1[0]] += upd
                # jump_utility_update[pth1[0]][prev_pth[-1]] += upd

            if _debug:
                print('proba_paths =', proba_paths)
        if _debug:
            print('move_utility_update')
            print(_format2d(move_utility_update))
            print('jump_utility_update')
            print(_format2d(jump_utility_update))

        # update pheromone and re-calculate transition probs (M-step)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                move_utility[i][j] = (
                    (1-evaporation) * move_utility[i][j]
                    + evaporation * move_utility_update[i][j]
                )
                jump_utility[i][j] = (
                    (1-evaporation) * jump_utility[i][j]
                    + evaporation * jump_utility_update[i][j]
                )
        if _debug:
            print('move_utility')
            print(_format2d(move_utility))
            print('jump_utility')
            print(_format2d(jump_utility))

        m_evidence = [
            [
                move_heuristic[i][j] * (move_utility[i][j] ** herdness)
                for j in range(n)
            ]
            for i in range(n)
        ]
        j_evidence = [
            [
                jump_heuristic[i][j] * (jump_utility[i][j] ** herdness)
                for j in range(n)
            ]
            for i in range(n)
        ]

        # at each iteration we greedily compute current best solution
        # so callback can see the progress
        best_paths = _make_aco_proba(
            n, n_salesmans,
            m_evidence,
            j_evidence,
            greedy=True)
        callback(best_paths, move_utility, j_evidence)

    return best_paths


def _format1d(vector):
    return '[' + ', '.join('{:.3f}'.format(v) for v in vector) + ']'


def _format2d(matrix):
    return '\n'.join(map(_format1d, matrix))


def _make_aco_proba(n, n_salesmans, move_utility, jump_utility, greedy):
    not_visited = list(range(n))
    paths = [[0]]
    not_visited.remove(paths[0][0])
    jumps_left = n_salesmans - 1
    for _ in range(n-1):
        pth = paths[-1]
        c = pth[-1]
        moverow = [move_utility[c][i] for i in not_visited]
        jumprow = [jump_utility[c][i] for i in not_visited]
        jump_prob = jumps_left / len(not_visited)
        move_prob = 1 - jump_prob
        if greedy:
            jnc_p = _argmax(jumprow)
            mnc_p = _argmax(moverow)
            jump = jump_prob*jumprow[jnc_p] > move_prob*moverow[mnc_p]
        else:
            jnc_p = _weighted_choice(jumprow)
            mnc_p = _weighted_choice(moverow)

            jp = jump_prob * jumprow[jnc_p]
            mp = move_prob * moverow[mnc_p]
            jump = _weighted_choice([mp, jp])
        nc_p = jnc_p if jump else mnc_p
        nc = not_visited.pop(nc_p)
        del moverow[nc_p]
        del jumprow[nc_p]
        if jump:
            paths.append([nc])
            jumps_left -= 1
        else:
            pth.append(nc)
    assert len(paths) <= n_salesmans
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
    return [e/expsum for e in exps]


def _normalize(vector):
    vsum = sum(vector)
    return [v/vsum for v in vector]


def _path_len(path, distances):
    d = 0
    for i in range(len(path)):
        d += distances[path[i-1]][path[i]]
    return d


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

    def __call__(self, paths, *utilities):
        width = 500
        contents = """
        <?xml version="1.0" encoding="utf-8" ?>
        <svg width="{}" height="{}" version="1.1"
             baseProfile="full" xmlns="http://www.w3.org/2000/svg">
        """.lstrip().format(len(utilities) * width, width)
        for i, utility in enumerate(utilities):
            contents += self._render_graph(utility, paths, i * width)
            paths = []
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
