def sort_b1b2b3(b1b2b3):
    ind = []
    for b in b1b2b3:
        indices = [b.find('W'), b.find('Y'), b.find('R')]
        sorted_indices = sorted(range(len(indices)), key=lambda k: indices[k])
        ind.append(sorted_indices)
    return ind