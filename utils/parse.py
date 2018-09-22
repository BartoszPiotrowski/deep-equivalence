def parse(formula):
    if len(formula) == 1:
        return formula
    else:
        predicate = formula[0]
        args_raw = formula[2:-1]
        args = []
        counter = 0
        i = 0
        while args_raw[i:]:
            if args_raw[i] == '(':
                counter += 1
            if args_raw[i] == ')':
                counter -= 1
            if counter == 0 and args_raw[i] == ',':
                args.append(args_raw[:i])
                args_raw = args_raw[i:].strip(',')
                i = 0
            i += 1
        args.append(args_raw)
    return [predicate, [parse(a) for a in args]]

