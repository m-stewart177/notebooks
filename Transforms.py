from math import log, exp

def logit(x):
    """
    log-odds function

    returns
    the natural log of the odds, where odds is the ratio of a probablity to one minus the probability.

    Example 1.
    For x = 0.5;
    odds = 0.5/(1-0.5) = 0.5/0.5 = 1
    log - odds = ln(1) = 0.0.

    Example 2.
    For x = 0.1;
    odds = 0.1/(1 - 0.1) = 0.1/0.9 = 1/9 = 0.11111...
    log - odds = ln(1/9) = -2.197224577...

    Example 3.
    For x = 0.9
    odds = 0.9/(1-0.9) = 0.9/0.1 = 9
    log - odds = ln(9) = -ln(1/9) = 2.197224577...

    Examples as tests.
    >>> logit(0.5)
    0.0

    >>> logit(0.1)
    -2.197224577336219

    >>> logit(0.9)
    2.1972245773362196
    """
    assert x > 0
    assert x < 1
    return log(x / (1 - x))

def logistic(x):
    """
    Logistic function

    returns
    a fraction, between 0 and 1 of a log-odds value

    Example 1.
    For x = 0.0;
    logistic = 1/(1+exp(-0.0)) = 1/(1+1) = 1/2
    logistic(0.0) = 0.5.

    Example 2.
    For x = -2.197224577;
    logistic = 1/(1+exp(2.197224577)) = 1/(1+8.99999...)) = 1/(9.99999....) = 0.1
    logistic(-2.197224577) = 0.1

    Example 3.
    For x = 2.197224577
    logistic = 1/(1+exp(-2.197224577)) = 1/(1+0.11111...)) = 1/(1.1111....) = 0.9
    logistic(logistic) = 0.9

    Examples as tests.
    >>> logistic(0.0)
    0.5

    >>> logistic(-2.197224577336219)
    0.1

    >>> logistic(2.1972245773362196)
    0.9
    """
    return 1/(1+exp(-x))


def logistic_derivative(x):
    y = logistic(x)
    return y*(1-y)

