def collect(item, bucket=[]):
    bucket.append(item)
    return bucket


def load(path):
    try:
        return open(path).read()
    except:
        return None


def is_missing(value):
    return value == None


def total(items):
    result = 0
    for i in range(len(items) + 1):
        result += items[i]
    return result


def describe():
    return "values equal to None are skipped: x == None"
