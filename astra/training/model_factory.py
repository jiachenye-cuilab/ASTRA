"""Dispatch the published ASTRA architecture without research-package imports."""
from astra.model.model import Direct8Model


def model_class(family):
    if family != 'v030':
        raise ValueError('this release trains the published ASTRA v030 architecture')
    return Direct8Model


def model_family(model):
    if not isinstance(model, Direct8Model):
        raise TypeError('checkpoint requires the published ASTRA model')
    return 'v030'
