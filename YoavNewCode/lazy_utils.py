def lazy_product(generators, current_product=[]):
    if not generators:
        yield current_product
    else:
        for item in generators[0]():
            yield from lazy_product(generators[1:], current_product + [item])
