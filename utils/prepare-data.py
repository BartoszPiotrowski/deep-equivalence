import argparse

parser = argparse.ArgumentParser(
    description="Takes a file with quadruples from Chad and produce file with"
                "lines of the form: "
                "1(expr_1 = expr_2), expr_1, expr_2.")
parser.add_argument('filename', type=str, help="Path to file with quadruples.")
args = parser.parse_args()

quads = open(args.filename).read().split('% Quad ')
quads = [q.split('\n')[1:5] for q in quads[1:]]

examples = []
for q in quads:
    examples.append(('1',q[0],q[2]))
    examples.append(('1',q[1],q[3]))
    examples.append(('0',q[0],q[3]))
    examples.append(('0',q[1],q[2]))

for e in examples:
    print(' '.join(e))

