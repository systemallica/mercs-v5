# -*- coding: UTF-8 -*-
"""
mercs.algo.induction
--- - --- - --- - ---

This module takes care of induction of component models.

author:
    Elia Van Wolputte
copyright:
    Copyright 2017-2018 KU Leuven, DTAI Research Group.
license:
    Apache License, Version 2.0, see LICENSE for details.
"""

from sklearn.tree import *
from sklearn.ensemble import *

from ..utils.metadata import only_nominal_targ, only_numeric_targ
from ..utils.keywords import *

from ..utils.debug import debug_print

VERBOSITY = 0


# Algorithms
def base_ind_algo(metadata, settings, m_targ):
    """
    Initialize a model.

    This only means initialization, not training.
    So the only thing that happens is deciding which model is going to be
    trained in a next step. E.g.; DT, RF, etc.

    Here, we mainly verify whether the targets are grouped correctly,
    i.e., are they nominal/numeric etc.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary of the MERCS model
    settings: dict
        Settings dictionary of the MERCS model
    m_targ: list, shape (nb_targ,)
        List of the indices of the attributes that will be targets of the model
        that will be trained afterwards.

    Returns
    -------

    """

    assert isinstance(m_targ, list)

    nb_mod = len(m_targ)
    assert nb_mod > 0

    is_nominal = metadata['is_nominal']

    msg = "is_nominal in this model is: {}".format(is_nominal)
    debug_print(msg, V=VERBOSITY)

    m_list = [induce_model(settings, is_nominal, m_targ[i])
              for i in range(nb_mod)]

    return m_list


# Inducers
def induce_model(settings, is_nominal, m_targ):
    """
    Initialize classifier/regressor with correct settings

    Parameters
    ----------
    settings: dict
        Dictionary of settings
    is_nominal: list, shape (nb_attributes, )
        Boolean array that indicates whether or not attribute is nominal
    m_targ: list, shape (nb_target_attributes, )
        List that contains the indices of all the target attributes of the
        model that is being initialized
    Returns
    -------

    """

    if only_nominal_targ(is_nominal, m_targ):
        model = induce_clf(settings)
    elif only_numeric_targ(is_nominal, m_targ):
        model = induce_rgr(settings)
    else:
        msg = "Model with mixed targets {}".format(m_targ)
        raise TypeError(msg)

    return model


def induce_clf(s):
    """
    Induce a single classifier.

    Filters the parameters
    Initializes the actual model
    """

    mod_type = s['type']
    params = {k: v for k, v in s.items() if k not in {'type', 'flatten'}}

    if mod_type in kw_ind_trees():
        clf = DecisionTreeClassifier(**params)
    elif mod_type in kw_ind_forests():
        clf = RandomForestClassifier(**params)
    else:
        msg = "Did nog recognize classifier type: {}".format(mod_type)
        raise TypeError(msg)

    return clf


def induce_rgr(s):
    """
    Induce a single regressor.

    Filters the parameters
    Initializes the actual model
    """

    mod_type = s['type']
    params = {k: v for k, v in s.items() if not {'type', 'flatten'}}

    if mod_type in kw_ind_trees():
        rgr = DecisionTreeRegressor(**params)
    elif mod_type in kw_ind_forests():
        rgr = RandomForestRegressor(**params)
    else:
        msg = "Did not recognize regressor type: {}".format(mod_type)
        raise TypeError(msg)

    return rgr
