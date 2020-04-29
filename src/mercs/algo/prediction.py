# -*- coding: UTF-8 -*-
"""
mercs.algo.prediction
--- - --- - --- - ---

This module takes care of prediction in MERCS.

The various prediction algorithms are implemented in this module.

author:
    Elia Van Wolputte
copyright:
    Copyright 2017-2018 KU Leuven, DTAI Research Group.
license:
    Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import warnings

from ..utils.encoding import codes_to_query, encode_attribute

from ..utils.debug import debug_print

VERBOSITY = 0


# MI-pred
def mi_pred_algo(m_codes, q_codes):
    assert isinstance(m_codes, np.ndarray)
    assert isinstance(q_codes, np.ndarray)
    assert len(m_codes.shape) == len(q_codes.shape) == 2
    assert m_codes.shape[1] == q_codes.shape[1]

    # Preliminaries
    nb_mods, nb_atts, nb_qrys = _extract_global_numbers(m_codes, q_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    mas, aas = _init_mas_aas(nb_mods, nb_atts, nb_qrys)

    for q_idx in range(nb_qrys):
        mas[q_idx], aas[q_idx] = _mi_pred_qry(mas[q_idx],
                                              aas[q_idx],
                                              q_desc[q_idx],
                                              q_targ[q_idx],
                                              m_codes)

    return mas, aas


def _mi_pred_qry(mas, aas, q_desc, q_targ, m_codes):
    steps = [1]

    # Zero-step
    aas[q_desc] = 0

    for n in steps:
        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]

        # Activate atts/mods
        act_atts = _active_atts(q_targ)
        act_mods = _active_mods_mi(avl_atts, act_atts, avl_mods, avl_m_codes)

        aas[act_atts] = n
        mas[act_mods] = n

    return mas, aas


# MA-pred
def ma_pred_algo(m_codes, q_codes, settings):
    assert isinstance(m_codes, np.ndarray)
    assert isinstance(q_codes, np.ndarray)
    assert len(m_codes.shape) == len(q_codes.shape) == 2
    assert m_codes.shape[1] == q_codes.shape[1]

    initial_threshold = settings['param']
    step_size = settings['its']
    assert isinstance(initial_threshold, (int, float))
    assert isinstance(step_size, float)
    assert 0.0 < initial_threshold <= 1.0
    assert 0.0 < step_size < 1.0

    # Preliminaries
    nb_mods, nb_atts, nb_qrys = _extract_global_numbers(m_codes, q_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    thresholds = np.arange(initial_threshold, -1, -step_size)

    mas, aas = _init_mas_aas(nb_mods, nb_atts, nb_qrys)

    for q_idx in range(nb_qrys):
        mas[q_idx], aas[q_idx] = _ma_pred_qry(mas[q_idx],
                                              aas[q_idx],
                                              q_desc[q_idx],
                                              q_targ[q_idx],
                                              m_codes,
                                              thresholds)
        pass

    return mas, aas


def _ma_pred_qry(mas, aas, q_desc, q_targ, m_codes, thresholds):
    steps = [1]

    # Zero-step
    aas[q_desc] = 0

    mas_mi, _ = _mi_pred_qry(mas, aas, q_desc, q_targ, m_codes)
    mas[mas_mi == -1] = 0
    mas[mas_mi == 1] = -1

    for n in steps:
        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]

        # Activate atts/mods
        act_atts = _active_atts(q_targ)
        aas[act_atts] = n

        for att in act_atts:
            single_act_att = np.array([att])
            act_mods = _active_mods_ma(avl_atts,
                                       single_act_att,
                                       avl_mods,
                                       avl_m_codes,
                                       thresholds)
            mas[act_mods] = n

    return mas, aas


# MAFI-pred
def mafi_pred_algo(m_codes, q_codes, settings):
    assert isinstance(m_codes, np.ndarray)
    assert isinstance(q_codes, np.ndarray)
    assert len(m_codes.shape) == len(q_codes.shape) == 2
    assert m_codes.shape[1] == q_codes.shape[1]

    initial_threshold = settings['param']
    step_size = settings['its']
    assert isinstance(initial_threshold, (int, float))
    assert isinstance(step_size, float)
    assert 0.0 < initial_threshold <= 1.0
    assert 0.0 < step_size < 1.0

    # TODO: This must not come packed in 'settings'
    feature_importances = settings['FI']

    # Preliminaries
    nb_mods, nb_atts, nb_qrys = _extract_global_numbers(m_codes, q_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    thresholds = np.arange(initial_threshold, -1, -step_size)

    mas, aas = _init_mas_aas(nb_mods, nb_atts, nb_qrys)

    for q_idx in range(nb_qrys):
        mas[q_idx], aas[q_idx] = _mafi_pred_qry(mas[q_idx],
                                                aas[q_idx],
                                                q_desc[q_idx],
                                                q_targ[q_idx],
                                                m_codes,
                                                thresholds,
                                                feature_importances)

    return mas, aas


def _mafi_pred_qry(mas,
                   aas,
                   q_desc,
                   q_targ,
                   m_codes,
                   thresholds,
                   feature_importances):
    steps = [1]

    # Zero-step
    aas[q_desc] = 0

    mas_mi, _ = _mi_pred_qry(mas, aas, q_desc, q_targ, m_codes)
    mas[mas_mi == -1] = 0
    mas[mas_mi == 1] = -1

    for n in steps:
        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]
        avl_f_imprt = feature_importances[avl_mods]

        # Activate atts/mods
        act_atts = _active_atts(q_targ)
        aas[act_atts] = n

        for att in act_atts:
            single_act_att = np.array([att])
            act_mods = _active_mods_mafi(avl_atts,
                                         single_act_att,
                                         avl_mods,
                                         avl_m_codes,
                                         thresholds,
                                         avl_f_imprt)
            mas[act_mods] = n

    return mas, aas


# IT-pred
def it_pred_algo(m_codes, q_codes, settings):
    assert isinstance(m_codes, np.ndarray)
    assert isinstance(q_codes, np.ndarray)
    assert len(m_codes.shape) == len(q_codes.shape) == 2
    assert m_codes.shape[1] == q_codes.shape[1]

    initial_threshold = 1.0
    step_size = settings['param']
    max_layers = settings['its']
    assert isinstance(max_layers, int)
    assert isinstance(step_size, float)
    assert 1 <= max_layers
    assert 0.0 < step_size < 1.0

    feature_importances = settings['FI']  # TODO: This must not come packed in 'settings'

    # Preliminaries
    nb_mods, nb_atts, nb_qrys = _extract_global_numbers(m_codes, q_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    thresholds = np.arange(initial_threshold, -1, -step_size)
    steps = list(range(1, max_layers + 1))

    mas, aas = _init_mas_aas(nb_mods, nb_atts, nb_qrys)

    for q_idx in range(nb_qrys):
        mas[q_idx], aas[q_idx] = _it_pred_qry(mas[q_idx],
                                              aas[q_idx],
                                              q_desc[q_idx],
                                              q_targ[q_idx],
                                              m_codes,
                                              thresholds,
                                              feature_importances,
                                              steps)

    return mas, aas


def _it_pred_qry(mas,
                 aas,
                 q_desc,
                 q_targ,
                 m_codes,
                 thresholds,
                 feature_importances,
                 steps):
    # Zero-step
    aas[q_desc] = 0

    done = False
    for n in steps:
        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]
        avl_f_imprt = feature_importances[avl_mods]

        msg = """
        AFTER COLLECTION OF AVAILABLE STUFF
        step: {}\n
        avl_atts: {}\n
        avl_mods: {}\n
        avl_m_codes: {}\n
        """.format(n, avl_atts, avl_mods, avl_m_codes)
        debug_print(msg, V=VERBOSITY)

        # Activate models
        pot_act_atts = _unavailable_atts(aas)
        act_mods = _active_mods_mafi(avl_atts,
                                     pot_act_atts,
                                     avl_mods,
                                     avl_m_codes,
                                     thresholds,
                                     avl_f_imprt,
                                     mode='some')
        mas[act_mods] = n

        msg = """
        AFTER MODEL ACTIVATION: \n
        unavl_atts: {}\n
        act_mods: {}\n
        mas: {}\n
        """.format(pot_act_atts, act_mods, mas)
        debug_print(msg, V=VERBOSITY)

        # Activate attributes
        act_m_codes = m_codes[act_mods]
        act_atts = _active_atts_it(act_m_codes, pot_act_atts)
        aas[act_atts] = n

        msg = """
        AFTER ATTRIBUTE ACTIVATION: \n
        act_atts: {}\n
        aas: {}\n
        """.format(act_atts, aas)
        debug_print(msg, V=VERBOSITY)

        # Assert whether we are done
        done = _assert_activation(aas, q_targ)
        if done:
            msg = """
            MAIN LOOP JUDGED DONE
            """
            debug_print(msg, V=VERBOSITY)
            break

    # If you are not done, repeat the last step until you are.
    n = steps[-1]
    while not done:
        msg = """
        Entering extra step to ensure finishing.
        """
        debug_print(msg, V=VERBOSITY)

        # Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]
        avl_f_imprt = feature_importances[avl_mods]

        # Activate models
        pot_act_atts = _unavailable_atts(aas)
        act_mods = _active_mods_mafi(avl_atts,
                                     pot_act_atts,
                                     avl_mods,
                                     avl_m_codes,
                                     thresholds,
                                     avl_f_imprt,
                                     mode='some')
        mas[act_mods] = n

        msg = """
        Active models: {}\n
        MAS at this point: {}\n
        """.format(act_mods, mas)
        debug_print(msg, V=VERBOSITY)

        # Activate attributes
        act_m_codes = m_codes[act_mods]
        act_atts = _active_atts_it(act_m_codes, pot_act_atts)
        aas[act_atts] = n

        msg = """
        Active atts: {}\n
        AAS at this point: {}\n
        """.format(act_atts, aas)
        debug_print(msg, V=VERBOSITY)

        # Assert whether we are done
        done = _assert_activation(aas, q_targ)

    return mas, aas


# RW-pred
def rw_pred_algo(m_codes, q_codes, settings):
    assert isinstance(m_codes, np.ndarray)
    assert isinstance(q_codes, np.ndarray)
    assert len(m_codes.shape) == len(q_codes.shape) == 2
    assert m_codes.shape[1] == q_codes.shape[1]

    max_layers = settings['its']
    assert isinstance(max_layers, int) and 1 <= max_layers
    chain_size = np.random.randint(1, max_layers + 1)  # Chain size at least one

    feature_importances = settings['FI']  # TODO: This must not come packed in 'settings'

    # Preliminaries
    nb_mods, nb_atts, nb_qrys = _extract_global_numbers(m_codes, q_codes)
    q_desc, q_targ, _ = codes_to_query(q_codes)

    steps = list(range(1, 1 + chain_size))  # Chain size at least one (it is correct that this +1 happens twice!)
    steps.reverse()

    mas, aas = _init_mas_aas(nb_mods, nb_atts, nb_qrys)

    for q_idx in range(nb_qrys):
        mas[q_idx], aas[q_idx] = _rw_pred_qry(mas[q_idx],
                                              aas[q_idx],
                                              q_desc[q_idx],
                                              q_targ[q_idx],
                                              m_codes,
                                              feature_importances,
                                              steps)

    return mas, aas


def _rw_pred_qry(mas,
                 aas,
                 q_desc,
                 q_targ,
                 m_codes,
                 feature_importances,
                 steps):
    desc_encoding = encode_attribute(0, [0], [1])

    # Zero-step
    aas[q_desc] = 0

    assert isinstance(steps, list)
    assert len(steps) >= 1

    for i, n in enumerate(steps):

        # 1. Collect available atts/mods
        avl_atts = _available_atts(aas, n)
        avl_mods = _available_mods(mas)

        avl_m_codes = m_codes[avl_mods]
        avl_f_imprt = feature_importances[avl_mods]

        msg = """
        After collection of available attributes and models\n
        --- - --- - --- - --- -- -- - --- - --- - --- - ---\n
        i, step:        {}, {}\n
        avl_atts:       {}\n
        avl_mods:       {}\n
        avl_m_codes:    {}\n
        """.format(i, n, avl_atts, avl_mods, avl_m_codes)
        debug_print(msg, level=2, V=VERBOSITY)

        # 2. Collection of potential active targets
        if i == 0:
            pot_act_atts = _active_atts(q_targ)
            assert pot_act_atts.shape[0] == 1  # TODO: multi-target RW
        else:
            unavailable_atts = _unavailable_atts(aas)
            next_mods = np.where(mas == n + 1)
            desc_atts_next_mods = np.where(m_codes[next_mods] == desc_encoding)[1]
            desc_atts_next_mods = np.unique(desc_atts_next_mods)
            pot_act_atts = np.intersect1d(unavailable_atts, desc_atts_next_mods)

        if len(pot_act_atts) == 0:
            break  # Nothing left to contribute

        # 3. Model activation
        act_mods = _active_mods_rw(avl_atts,
                                   pot_act_atts,
                                   avl_mods,
                                   avl_m_codes,
                                   avl_f_imprt)

        if len(act_mods) == 0:
            break  # Nothing left to contribute

        mas[act_mods] = n

        msg = """
        After model activation \n
        --- - --- - --- - --- -\n
        pot_act_atts:   {}\n
        act_mods:       {}\n
        mas:            {}\n
        """.format(pot_act_atts, act_mods, mas)
        debug_print(msg, level=2, V=VERBOSITY)

        # 3. Activate attributes
        act_m_codes = m_codes[act_mods]
        act_atts = _active_atts_it(act_m_codes, pot_act_atts)

        assert act_atts.shape[0] > 0
        aas[act_atts] = n

        msg = """
        After attribute activation \n
        --- - --- - --- - --- -\n
        act_atts: {}\n
        aas: {}\n
        """.format(act_atts, aas)
        debug_print(msg, level=2, V=VERBOSITY)

    assert np.max(aas) == np.max(mas)
    assert np.max(mas) >= 1

    mas, aas = recode_strat(mas, aas)
    return mas, aas


# (Un)available Attributes
def _available_atts(aas, step):
    """
    Available attributes were 'active' in a previous step.

    I.e.:

    aas == 0:
        Attribute was always available, i.e., descriptive attribute
        of the query.
    1 <= aas < step:
        Attribute was activated in a previous step of the prediction
        algorithm, and thus available

    Parameters
    ----------
    aas
    step

    Returns
    -------

    """
    return np.where((0 <= aas) & (aas < step))[0]


def _unavailable_atts(aas):
    return np.where(aas == -1)[0]


# (Un)available Models
def _available_mods(mas):
    """
    Available models are models that have not been 'active' before

    Parameters
    ----------
    mas

    Returns
    -------

    """
    return np.where(mas == -1)[0]


def _unavailable_mods(mas, step):
    return np.where((0 <= mas) & (mas < step))[0]


# Activation of Attributes
def _active_atts(q_targ):
    return np.array(q_targ)


def _active_atts_it(act_m_codes, unavl_atts):
    assert act_m_codes.shape[0] >= 1
    assert unavl_atts.shape[0] >= 1
    targ_encoding = encode_attribute(1, [0], [1])

    targ_atts_act_mods = np.where(act_m_codes == targ_encoding)[1]
    targ_atts_act_mods = np.unique(targ_atts_act_mods)

    unavl_atts_as_targ = np.intersect1d(unavl_atts, targ_atts_act_mods)

    return unavl_atts_as_targ


# Activation of Models
def _active_mods_mi(avl_atts, act_atts, avl_mods, avl_m_codes, strict=True):
    """
    Activate models corresponding to the MI-strategy.

    This means that of the avl_mods, we pass the entries that correspond
    to a model that has at least one of the active attributes in its target set.

    Furthermore, we ensure that any active attribute occurs at least once in a
    target set of attributes.

    Parameters
    ----------
    avl_atts
    act_atts
    avl_mods
    avl_m_codes

    Returns
    -------

    """

    act_mods_idx = _filter_mods_act_atts_as_targ(avl_mods, avl_m_codes, act_atts)

    if strict:
        assert _assert_all_act_atts_as_targ(act_mods_idx, avl_m_codes, act_atts)
    else:
        assert act_mods_idx.shape[0] >= 1

    act_mods = avl_mods[act_mods_idx]

    return act_mods


def _active_mods_ma(avl_atts, act_atts, avl_mods, avl_m_codes, thresholds):
    assert avl_m_codes.shape[0] == avl_mods.shape[0]
    desc_encoding = encode_attribute(0, [0], [1])

    # Calculate appropriateness scores for all available models
    avl_mods_appr_scores = np.zeros(avl_mods.shape[0])
    for m_idx, m_code in enumerate(avl_m_codes):
        overlap_avl_desc_atts = np.sum(m_code[avl_atts] == desc_encoding)
        total_count_desc_atts = np.sum(m_code[:] == desc_encoding)

        avl_mods_appr_scores[m_idx] = overlap_avl_desc_atts / total_count_desc_atts

    # Activate models with sufficiently high appropriateness scores
    for threshold in thresholds:
        act_mods_idx = np.where(avl_mods_appr_scores >= threshold)[0]

        if _assert_all_act_atts_as_targ(act_mods_idx, avl_m_codes, act_atts):
            break

    act_mods = avl_mods[act_mods_idx]

    return act_mods


def _active_mods_mafi(avl_atts,
                      act_atts,
                      avl_mods,
                      avl_m_codes,
                      thresholds,
                      avl_f_imprt,
                      mode='all'):
    assert avl_m_codes.shape[0] == avl_f_imprt.shape[0] == avl_mods.shape[0]

    # Calculate appropriateness scores for all available models
    avl_mods_appr_scores = np.zeros(avl_mods.shape[0])
    for m_idx in range(avl_mods.shape[0]):
        avl_mods_appr_scores[m_idx] = np.sum(avl_f_imprt[np.ix_([m_idx], avl_atts)])

    # Activate models with sufficiently high appropriateness scores
    if mode in {'all'}:
        stopping_criterion = _assert_all_act_atts_as_targ
    elif mode in {'some'}:
        stopping_criterion = _assert_some_act_atts_as_targ
    else:
        msg = """
        Did not recognize the mode: {}
        Choose either 'all' or 'some'.
        """.format(mode)
        raise ValueError(msg)

    for threshold in thresholds:
        act_mods_idx = np.where(avl_mods_appr_scores >= threshold)[0]

        if stopping_criterion(act_mods_idx, avl_m_codes, act_atts):
            break

    act_mods = avl_mods[act_mods_idx]

    assert act_mods.shape[0] > 0
    return act_mods


def _active_mods_rw(avl_atts,
                    act_atts,
                    avl_mods,
                    avl_m_codes,
                    avl_f_imprt):
    assert avl_m_codes.shape[0] == avl_f_imprt.shape[0] == avl_mods.shape[0]

    act_mods_idx = _filter_mods_act_atts_as_targ(avl_mods, avl_m_codes, act_atts)
    avl_mods = avl_mods[act_mods_idx]
    avl_m_codes = avl_m_codes[act_mods_idx]
    avl_f_imprt = avl_f_imprt[act_mods_idx]

    assert avl_m_codes.shape[0] == avl_f_imprt.shape[0] == avl_mods.shape[0]

    # Calculate appropriateness scores for all available models
    avl_mods_appr_scores = np.zeros(avl_mods.shape[0])
    for m_idx in range(avl_mods.shape[0]):
        avl_mods_appr_scores[m_idx] = np.sum(avl_f_imprt[np.ix_([m_idx], avl_atts)])

    act_mods_idx = pick_random_models_from_appr_score(avl_mods_appr_scores, n=1)

    act_mods = avl_mods[act_mods_idx]

    assert act_mods.shape[0] > 0  # Important assertion
    return act_mods


# Assert solution
def _assert_all_act_atts_as_targ(act_mods_idx, avl_m_codes, act_atts):
    if act_mods_idx.shape[0] == 0:
        return False
    else:
        target_encoding = encode_attribute(1, [0], [1])

        filtered_m_codes = avl_m_codes[np.ix_(act_mods_idx, act_atts)]
        act_atts_as_targ = np.where(filtered_m_codes == target_encoding)[1]
        check = np.unique(act_atts_as_targ)

        return check.shape[0] == act_atts.shape[0]


def _assert_some_act_atts_as_targ(act_mods_idx, avl_m_codes, act_atts):
    if act_mods_idx.shape[0] == 0:
        return False
    else:
        target_encoding = encode_attribute(1, [0], [1])

        filtered_m_codes = avl_m_codes[np.ix_(act_mods_idx, act_atts)]
        act_atts_as_targ = np.where(filtered_m_codes == target_encoding)[1]
        check = np.unique(act_atts_as_targ)

        return check.shape[0] >= 1


def _assert_activation(aas, targ):
    unavl_atts = _unavailable_atts(aas)
    return np.isin(targ, unavl_atts, invert=True).all()


# Helpers
def _filter_mods_act_atts_as_targ(avl_mods, avl_m_codes, act_atts):
    assert avl_m_codes.shape[0] == avl_mods.shape[0]
    targ_encoding = encode_attribute(1, [0], [1])

    # Calculate appropriateness scores for all available models
    avl_mods_appr_scores = np.zeros(avl_mods.shape[0])
    for m_idx, m_code in enumerate(avl_m_codes):
        avl_mods_appr_scores[m_idx] = np.sum(m_code[act_atts] == targ_encoding)

    # Activate models with sufficiently high appropriateness scores
    act_mods_idx = np.where(avl_mods_appr_scores >= 1.0)[0]

    return act_mods_idx


def pick_random_models_from_appr_score(mod_appr_scores, n=1):
    """
    Interpret an array of appropriateness scores as a multinominal distribution
    corresponding to the probability of a certain model being selected.


    Parameters
    ----------
    mod_appr_scores
    n

    Returns
    -------

    """
    norm = np.linalg.norm(mod_appr_scores, 1)
    if norm > 0:
        mod_appr_scores = mod_appr_scores / norm
    else:
        msg = """
        Not a single appropriate model was found, therefore
        making an arbitrary choice.
        """
        warnings.warn(msg)
        # If you cannot be right, be arbitrary
        mod_appr_scores = [1 / len(mod_appr_scores) for i in mod_appr_scores]

    picks = np.where(np.random.multinomial(n, mod_appr_scores, size=1)[0] == 1)

    return picks


def full_prune_strat(m_codes, q_code, mas, aas):
    """
    Prune the last step of the activation strategies.

    :param m_codes:         model codes
    :param q_code:          code of the queries
    :param mas:             Model Activation Strategy
    :param aas:             Attribute Activation Strategy
    :return:
    """

    max_step = np.max(aas)

    for i in range(max_step):
        step = max_step - i

        if step == max_step:
            aas = np.array([e if ((e != step) | (q_code[i] == 1))
                            else 0 for i, e in enumerate(aas)])
        else:
            next_act_mods = (mas > step)

            msg = """
            Next act mods:\t{}\n
            """.format(next_act_mods)
            debug_print(msg, V=VERBOSITY)

            aas = np.array(
                [e if ((e != step) | (np.sum(m_codes[next_act_mods, i] == 0) > 0) | (q_code[i] == 1))
                 else 0 for i, e in enumerate(aas)])

        act_atts = (aas >= step)
        mas = [e if ((e != step) | (np.sum(m_codes[i][act_atts] == 1) > 0))
               else 0 for i, e in enumerate(mas)]

    # Recode strategies
    mas, aas = recode_strat(mas, aas)

    return mas, aas


def recode_strat(mas, aas):
    """
    Avoid gaps in strategy, and make sure that it starts from one.

    :param mas:     Model Activation Strategy
    :param aas:     Attribute Activation Strategy
    :return:
    """

    # Recode the arrays, some layers might disappear completely!
    uniq_mas, uniq_aas = np.unique(mas), np.unique(aas)
    uniq_mas = np.array([e for e in uniq_mas if e > 0])
    uniq_aas = np.array([e for e in uniq_aas if e > 0])
    mas = np.array([np.where(uniq_mas == e)[0][0] + 1 if e > 0 else e for e in mas])
    aas = np.array([np.where(uniq_aas == e)[0][0] + 1 if e > 0 else e for e in aas])  # -1 for unknown targets

    return mas, aas


# Initialize
def _extract_global_numbers(m_codes, q_codes):
    nb_mods, nb_atts = m_codes.shape
    nb_qrys = q_codes.shape[0]
    return nb_mods, nb_atts, nb_qrys


def _init_mas_aas(nb_models, nb_atts, nb_queries):
    """
    Initialize the mas and aas arrays.

    Difference with the above is that this outputs full numpy arrays.

    Parameters
    ----------
    nb_models: int
        Total number of models in the MERCS model
    nb_atts: int
        Total number of attributes in the MERCS model
    nb_queries: int
        Total number of queries that needs to be answered.

    Returns
    -------
    mas: np.ndarray, shape (nb_queries, nb_models)
        E.g.:
            np.array([[0,0,0],
                      [0,0,0]])
    aas: np.ndarray, shape (nb_queries, nb_atts)
        E.g.:
            np.array([[-1,-1,-1,-1],
                      [-1,-1,-1,-1]])
    """

    assert isinstance(nb_models, int)
    assert isinstance(nb_atts, int)
    assert isinstance(nb_queries, int)

    mas = np.full((nb_queries, nb_models), -1, dtype=np.int)
    aas = np.full((nb_queries, nb_atts), -1, dtype=np.int)
    return mas, aas
