import numpy


def normalize(x: numpy.ndarray):
   
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


def standardize(x: numpy.ndarray):
   
    return (x - x.mean(axis=0)) / numpy.std(x, axis=0)
