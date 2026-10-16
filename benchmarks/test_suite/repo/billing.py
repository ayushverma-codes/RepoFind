# TODO: support multiple currencies


def price(amount, rate):
    return amount * rate


def discount(amount):
    # FIXME: discount tiers are hard coded
    return amount * 0.9
