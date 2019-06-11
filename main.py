import argparse
import csv
import sys
import pathlib
import math
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

    heuristic = tsp_aco.HeuristicV1(
        distances=distmat,
        n_salesmans=opts.m,
        herdness=0.5,
        greedness=0.5,
        evaporation=0.01,
        n_probas=opts.n_probas or int(opts.n * math.log(opts.n)) * opts.m,
        features=opts.features.split(","),
    )
    best_paths = tsp_aco.solve_tsp(
        distances=distmat,
        callback=tsp_aco.GenerateVisualSvg("frames", coordinates),
        max_iteration=opts.iterations or int(opts.n * math.log(opts.n)),
        heuristic=heuristic,
    )
    for path in best_paths:
        cost = tsp_aco.path_len(path, distmat)
        print(cost, "::", " -> ".join(city_names[i] for i in path))

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


# main('-n 10 -m 1 -p 20 -I 100'.split())
if __name__ == "__main__":
    main()
