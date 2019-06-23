import argparse
import csv
import sys
import pathlib
import math
import os
import webbrowser
import tsp_aco
from graph import parse_graph_from_file


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-F", default="worldcitiespop-part1.txt")
    parser.add_argument("--country", "-C", default="hu")
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("-m", type=int, default=2)
    parser.add_argument("--iterations", "-I", type=int)
    parser.add_argument("--n-probas", "-p", type=int)
    parser.add_argument("--features", default="")
    parser.add_argument("--out", "-O")
    parser.add_argument(
        "--format", "-f", choices=["readable", "csv", "numpy"], default="csv"
    )
    parser.add_argument("--show", "-S", action="store_true")
    opts = parser.parse_args(argv)

    city_names, coordinates, distmat = parse_graph_from_file(
        opts.file, opts.country, opts.n
    )

    if opts.out == "-":
        out = sys.stdout
    elif opts.out:
        out = open(opts.out, "w")
    else:
        out = None
    if out is not None:
        printers = {"readable": print_readable, "csv": print_csv, "numpy": print_numpy}
        printers[opts.format](out, city_names, distmat)

        if opts.out != "-":
            out.close()

    graph_name = os.path.split(opts.file)[1].rpartition(".")[0]

    # heuristic = tsp_aco.HeuristicV1(
    #     distances=distmat,
    #     n_salesmans=opts.m,
    #     herdness=0.5,
    #     greedness=0.5,
    #     evaporation=0.001,
    #     n_probas=1000,
    #     use_jump_balancing=True,
    #     pheromone_update_scale=1,
    # )
    heuristic = tsp_aco.HeuristicV2(
        distances=distmat,
        n_salesmans=opts.m,
        herdness=0.5,
        greedness=1,
        evaporation=0.01,
        # n_probas=int(opts.n * math.log(opts.n)) * opts.m,
        n_probas=2000,
        pheromone_update_scale=0.0001,
        use_jump_balancing=False,
    )
    best_cost, best_paths = tsp_aco.solve_tsp(
        distances=distmat,
        callback=tsp_aco.GenerateVisualSvg(f"frames/test", coordinates),
        # callback=tsp_aco.GenerateVisualSvg(f"frames/v2_{graph_name}", coordinates),
        max_iteration=20,
        heuristic=heuristic,
    )
    # distmat_norm = sum(map(sum, distmat)) / (len(distmat) ** 2)
    for path in best_paths:
        cost = tsp_aco.path_len(path, distmat) #/ distmat_norm
        print(f"{cost:.4f}:: {' -> '.join(str(i) for i in path)}")

    if opts.show:
        url = (
            pathlib.Path(__file__)
            .absolute()
            .parent.joinpath("visualizer.html")
            .as_uri()
        )
        webbrowser.open(url)


def print_readable(out, names, distmap):
    printmat = [["", *names]]
    for name, row in zip(names, distmap):
        printmat.append([name, *("{:.2f}".format(d) for d in row)])
    colw = max(map(len, (x for r in printmat for x in r)))
    for row in printmat:
        out.write(" ".join(r.ljust(colw) for r in row))
        out.write("\n")


def print_csv(out, names, distmap):
    writer = csv.writer(out)
    writer.writerow(names)
    for line in distmap:
        writer.writerow(map(str, line))


def print_numpy(out, names, distmap):
    import numpy

    numpy.savetxt(out, distmap)


main(
    "-C x -F graphs/1-30,gauss,1.csv "
    "-n 30 -m 3 -p 1000 -I 5 "
    # "--features JUMP_BALANCING"
    "".split()
)
# if __name__ == "__main__":
#     main()
