import os
import sys
import warnings
from os.path import dirname

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score

import datasets as datasets
from mercs.core import MERCS
from mercs.utils.encoding import encode_attribute

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# Custom import (Add src to the path)
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))


def setup_classification():
    train, test = datasets.load_nursery()
    model = MERCS()

    ind_parameters = {'ind_type': 'RF',
                      'ind_n_estimators': 10,
                      'ind_max_depth': 4}

    sel_parameters = {'sel_type': 'Base',
                      'sel_its': 4,
                      'sel_param': 2}

    model.fit(train, **ind_parameters, **sel_parameters)

    code = [0, 0, 0, 0, 0, 0, 0, 0, 1]

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

    obs = f1_score(y_true, y_pred, average='macro')

    assert isinstance(obs, (int, float))
    assert 0 <= obs <= 1


def test_MA_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type': 'MA',
                       'pred_param': 0.95,
                       'pred_its': 0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs = f1_score(y_true, y_pred, average='macro')

    assert isinstance(obs, (int, float))
    assert 0 <= obs <= 1


def test_MAFI_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type': 'MAFI',
                       'pred_param': 0.95,
                       'pred_its': 0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs = f1_score(y_true, y_pred, average='macro')

    assert isinstance(obs, (int, float))
    assert 0 <= obs <= 1


def test_IT_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type': 'IT',
                       'pred_param': 0.1,
                       'pred_its': 8}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs = f1_score(y_true, y_pred, average='macro')

    assert isinstance(obs, (int, float))
    assert 0 <= obs <= 1


def test_RW_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type': 'RW',
                       'pred_param': 2,
                       'pred_its': 16}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs = f1_score(y_true, y_pred, average='macro')

    assert isinstance(obs, (int, float))
    assert 0 <= obs <= 1
