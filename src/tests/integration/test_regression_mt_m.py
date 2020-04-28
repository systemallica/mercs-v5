import os
import sys
import warnings
from os.path import dirname

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_squared_log_error)

from mercs.core import MERCS
from mercs.utils.encoding import encode_attribute
import datasets as datasets

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# Custom import (Add src to the path)
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))


def setup_regression():
    train, test = datasets.load_slump()
    model = MERCS()

    ind_parameters = {'ind_type': 'RF',
                      'ind_n_estimators': 10,
                      'ind_max_depth': 4}

    sel_parameters = {'sel_type': 'Base',
                      'sel_its': 4,
                      'sel_param': 2}

    model.fit(train, **ind_parameters, **sel_parameters)

    code = [-1, -1, 0, 0, 0, 0, 0, 0, 1, 1]

    target_boolean = np.array(code) == encode_attribute(2, [1], [2])
    y_true = test[test.columns.values[target_boolean]].values
    return train, test, code, model, y_true


def test_MI_regression():
    train, test, code, model, y_true = setup_regression()

    pred_parameters = {'pred_type': 'MI',
                       'pred_param': 0.95,
                       'pred_its': 0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs_1 = mean_absolute_error(y_true, y_pred)
    obs_2 = mean_squared_error(y_true, y_pred)
    obs_3 = mean_squared_log_error(y_true, y_pred)

    obs = [obs_1, obs_2, obs_3]

    for o in obs:
        assert isinstance(o, (int, float))
        assert 0 <= o

    return


def test_MA_regression():
    train, test, code, model, y_true = setup_regression()

    pred_parameters = {'pred_type': 'MA',
                       'pred_param': 0.95,
                       'pred_its': 0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs_1 = mean_absolute_error(y_true, y_pred)
    obs_2 = mean_squared_error(y_true, y_pred)
    obs_3 = mean_squared_log_error(y_true, y_pred)

    obs = [obs_1, obs_2, obs_3]

    for o in obs:
        assert isinstance(o, (int, float))
        assert 0 <= o
    return


def test_MAFI_regression():
    train, test, code, model, y_true = setup_regression()

    pred_parameters = {'pred_type': 'MAFI',
                       'pred_param': 0.95,
                       'pred_its': 0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs_1 = mean_absolute_error(y_true, y_pred)
    obs_2 = mean_squared_error(y_true, y_pred)
    obs_3 = mean_squared_log_error(y_true, y_pred)

    obs = [obs_1, obs_2, obs_3]

    for o in obs:
        assert isinstance(o, (int, float))
        assert 0 <= o
    return


def test_IT_regression():
    train, test, code, model, y_true = setup_regression()

    pred_parameters = {'pred_type': 'IT',
                       'pred_param': 0.1,
                       'pred_its': 8}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs_1 = mean_absolute_error(y_true, y_pred)
    obs_2 = mean_squared_error(y_true, y_pred)
    obs_3 = mean_squared_log_error(y_true, y_pred)

    obs = [obs_1, obs_2, obs_3]

    for o in obs:
        assert isinstance(o, (int, float))
        assert 0 <= o
    return


# TODO: RW does not do muli-target.
"""
def test_RW_classification():
    train, test, code, model, y_true = setup_classification()

    pred_parameters = {'pred_type':     'RW',
                       'pred_param':    2,
                       'pred_its':      16}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    obs_1 = mean_absolute_error(y_true, y_pred)
    obs_2 = mean_squared_error(y_true, y_pred)
    obs_3 = mean_squared_log_error(y_true, y_pred)

    obs = [obs_1, obs_2, obs_3]

    for o in obs:
        assert isinstance(o, (int, float))
        assert 0 <= o
    return
"""
