def matches(filter, value):
    return any(x in value for x in filter)


def filter(dict, filt):
    return {k: v for k, v in dict.items() if matches(filt, v)}
