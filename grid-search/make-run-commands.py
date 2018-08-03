if __name__ == "__main__":
    import argparse
    import datetime
    import itertools
    import os
    import re

    COMMAND = 'python3'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "grid",
        type=str,
        help="Path to a file with parameters grid of form: \
                param_1 val_1 val_2 val_3 \
                param_2 val_1 val_2 \
                ... \
        .")
    args = parser.parse_args()
    params = []
    values = []
    with open(args.grid, 'r') as grid_file:
        for line in grid_file:
            p = line.strip('\n').split(' ')
            params.append(p[0])
            values.append(p[1:])
    product = itertools.product(*values)
    for v in product:
        to_run = COMMAND + ' ' + ' '.join([' '.join(p) for p in zip(params, v)])
        print(to_run)

