import numpy as np

from sklearn.impute import SimpleImputer

from mercs.algo.inference import perform_imputation
from mercs.utils.encoding import encode_attribute
from datasets.datasets import load_nursery


def test_perform_imputation():
    # Prelims
    train, test = load_nursery()
    query_code = np.array([0, -1, -1, -1, -1, -1, 0, 0, 1])

    imputator = SimpleImputer(strategy='most_frequent')
    imputator.fit(train)

    # Actual test
    obs = perform_imputation(test, query_code, imputator)

    assert test.shape == obs.shape
    assert isinstance(obs, np.ndarray)

    boolean_missing = encode_attribute(0, [1], [2])

    for row in obs[:, boolean_missing].T:
        assert len(np.unique(row)) == 1
