import numpy as np
from sklearn.metrics import f1_score

import datasets as ds
from mercs.core import MERCS
from mercs.utils.encoding import encode_attribute


def setup_classification():
    train, test = ds.load_nursery()
    model = MERCS()

    ind_parameters = {'ind_type': 'RF',
                      'ind_n_estimators': 10,
                      'ind_max_depth': 4}

    sel_parameters = {'sel_type': 'Base',
                      'sel_its': 4,
                      'sel_param': 2}

    model.fit(train, **ind_parameters, **sel_parameters)

    code = [-1, -1, -1, 0, 0, 0, 0, 1, 1]

    target_boolean = np.array(code) == encode_attribute(2, [1], [2])
    y_true = test[test.columns.values[target_boolean]].values
    return train, test, code, model, y_true


def test_MI_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type': 'MI',
                       'pred_param': 0.95,
                       'pred_its': 0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    nb_targets = y_true.shape[1]
    for t_idx in range(nb_targets):
        single_y_true = y_true[:][t_idx]
        single_y_pred = y_pred[:][t_idx]
        obs = f1_score(single_y_true, single_y_pred, average='macro')

        assert isinstance(obs, (int, float))
        assert 0 <= obs <= 1
    return


def test_MA_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type': 'MA',
                       'pred_param': 0.95,
                       'pred_its': 0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    nb_targets = y_true.shape[1]
    for t_idx in range(nb_targets):
        single_y_true = y_true[:][t_idx]
        single_y_pred = y_pred[:][t_idx]
        obs = f1_score(single_y_true, single_y_pred, average='macro')

        assert isinstance(obs, (int, float))
        assert 0 <= obs <= 1
    return


def test_MAFI_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type': 'MAFI',
                       'pred_param': 0.95,
                       'pred_its': 0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    nb_targets = y_true.shape[1]
    for t_idx in range(nb_targets):
        single_y_true = y_true[:][t_idx]
        single_y_pred = y_pred[:][t_idx]
        obs = f1_score(single_y_true, single_y_pred, average='macro')

        assert isinstance(obs, (int, float))
        assert 0 <= obs <= 1
    return


def test_IT_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type': 'IT',
                       'pred_param': 0.1,
                       'pred_its': 8}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    nb_targets = y_true.shape[1]
    for t_idx in range(nb_targets):
        single_y_true = y_true[:][t_idx]
        single_y_pred = y_pred[:][t_idx]
        obs = f1_score(single_y_true, single_y_pred, average='macro')

        assert isinstance(obs, (int, float))
        assert 0 <= obs <= 1
    return


"""
def test_RW_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type':     'RW',
                       'pred_param':    2,
                       'pred_its':      16}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs = f1_score(y_true, y_pred, average='macro')

    assert isinstance(obs, (int, float))
    assert 0 <= obs <= 1
    return
"""
