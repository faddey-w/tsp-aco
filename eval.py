import tsp_aco
import random
import math


def generate_graph(n, form, scale, center=(0, 0)):
    if form == "cluster":

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


def merge(*graphs):
    return sum(graphs, [])


def main():
    graph_params = [(15, "cluster", 1)]
    solver_params = [
        dict(
            v=1,
            n_salesmans=1,
            greedness=0.5,
            herdness=0.5,
            evaporation=0.1,
            n_probas=10,
            max_iteration=20,
        ),
        dict(
            v=1,
            n_salesmans=2,
            greedness=0.5,
            herdness=0.5,
            evaporation=0.01,
            n_probas=10,
            max_iteration=20,
        ),
        dict(
            v=1,
            n_salesmans=2,
            greedness=0.5,
            herdness=0.5,
            evaporation=0.1,
            n_probas=30,
            max_iteration=20,
        ),
        dict(
            v=1,
            n_salesmans=2,
            greedness=2.0,
            herdness=2.0,
            evaporation=0.1,
            n_probas=10,
            max_iteration=20,
        ),
        dict(
            v=1,
            n_salesmans=2,
            greedness=2.0,
            herdness=2.0,
            evaporation=0.1,
            n_probas=10,
            max_iteration=0,
        )
    ]
    n_tests_per_setup = 100
    visualize = False

    for graph_p in graph_params:
        costs_acc = [0] * len(solver_params)
        for test_i in range(n_tests_per_setup):
            if isinstance(graph_p, tuple):
                graph = generate_graph(*graph_p)
            else:
                graph = merge(*(generate_graph(*p) for p in graph_p))
            n = len(graph)
            distances = [
                [
                    (
                        (graph[i][0] - graph[j][0]) ** 2
                        + (graph[i][1] - graph[j][1]) ** 2
                    )
                    ** 0.5
                    for j in range(n)
                ]
                for i in range(n)
            ]
            costs = []
            for solver_i, solver_p in enumerate(solver_params):
                params = dict(solver_p, distances=distances)
                version = params.pop("v")
                max_iteration = params.pop("max_iteration")
                if version == 1:
                    heuristic = tsp_aco.HeuristicV1(**params)
                elif version == 2:
                    heuristic = tsp_aco.HeuristicV2(**params)
                else:
                    raise ValueError
                c, _ = tsp_aco.solve_tsp(
                    distances=distances,
                    max_iteration=max_iteration,
                    heuristic=heuristic,
                    callback=tsp_aco.GenerateVisualSvg("frames", graph)
                    if visualize and test_i == 0
                    else None,
                )
                costs.append(c)
            max_cost = max(costs)
            costs_acc = [ca + c / max_cost for ca, c in zip(costs_acc, costs)]
            print(test_i, [c / max_cost for c in costs], costs, max_cost)
        print(graph_p, "\t".join(f"{c/n_tests_per_setup:.3f}" for c in costs_acc))


if __name__ == "__main__":
    main()
