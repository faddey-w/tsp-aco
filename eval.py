import tsp_aco
import random
import numpy as np
from graph import parse_graph_from_file


def main():
    graph_names = [
        "1-15,gauss,1",
        # "1-15,gauss,2",
        "777-15,gauss,1",
        # "777-15,gauss,2",
    ]
    solver_params = [
        # {
        #     "v": 1,
        #     "n_salesmans": 1,
        #     "greedness": 0.5,
        #     "herdness": 0.5,
        #     "evaporation": 0.1,
        #     "n_probas": 10,
        #     "max_iteration": 20,
        # },
        {
            "v": 1,
            "n_salesmans": 2,
            "greedness": 0.5,
            "herdness": 0.5,
            "evaporation": 0.01,
            "n_probas": 10,
            "max_iteration": 20,
        },
        {
            "v": 1,
            "n_salesmans": 2,
            "greedness": 0.5,
            "herdness": 0.5,
            "evaporation": 0.1,
            "n_probas": 30,
            "max_iteration": 20,
        },
        {
            "v": 1,
            "n_salesmans": 2,
            "greedness": 2.0,
            "herdness": 2.0,
            "evaporation": 0.1,
            "n_probas": 10,
            "max_iteration": 20,
        },
        {
            "v": 1,
            "n_salesmans": 2,
            "greedness": 2.0,
            "herdness": 2.0,
            "evaporation": 0.01,
            "n_probas": 10,
            "max_iteration": 20,
        },
        # {
        #     "v": 1,
        #     "n_salesmans": 2,
        #     "greedness": 2.0,
        #     "herdness": 2.0,
        #     "evaporation": 0.1,
        #     "n_probas": 10,
        #     "max_iteration": 0,
        # },
    ]
    n_tests_per_setup = 50

    results = np.zeros((len(graph_names), len(solver_params), n_tests_per_setup, 2))

    for graph_i, name in enumerate(graph_names):
        path = f"graphs/{name}.csv"
        _, _, distances = parse_graph_from_file(path)
        dist_norm = sum(map(sum, distances)) / (len(distances) ** 2)
        for test_i in range(n_tests_per_setup):
            for solver_i, solver_p in enumerate(solver_params):
                print(
                    f"graph: {graph_i}/{len(graph_names)}; "
                    f"test: {test_i}/{n_tests_per_setup}; "
                    f"solver: {solver_i}/{len(solver_params)}"
                )
                params = dict(solver_p, distances=distances)
                version = params.pop("v")
                max_iteration = params.pop("max_iteration")
                if version == 1:
                    heuristic = tsp_aco.HeuristicV1(**params)
                elif version == 2:
                    heuristic = tsp_aco.HeuristicV2(**params)
                else:
                    raise ValueError
                random.seed(test_i)
                c, _ = tsp_aco.solve_tsp(
                    distances=distances,
                    max_iteration=max_iteration,
                    heuristic=heuristic,
                    callback=None,
                )
                results[graph_i, solver_i, test_i] = c, c / dist_norm

    np.save("results.npy", results)


if __name__ == "__main__":
    main()
