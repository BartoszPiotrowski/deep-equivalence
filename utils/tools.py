


def one_hot(elem, elems):
    if isinstance(elems, int):
        assert 0 <= elem < elems
        elems = range(elems)
    else:
        assert len(set(elems)) == len(elems)
    return [1 if e ==  elem else 0 for e in elems]
