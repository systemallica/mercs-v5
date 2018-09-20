import numpy as np
import pandas as pd

# Debugger verbosity
from ..utils.debug import debug_print
VERBOSITY = 1


# Everything related to codes
def codes_to_query(codes, atts=None):
    """
    Change the codes-array to an actual queries, which are three arrays.

    :param code:                Array that contains: 0-desc/1-target/-1-missing code for each attribute
    :param atts:                Array that contains the attributes (indices)
    :return: Three arrays.      One for desc atts indices, one for targets, one for missing
    """

    if atts is None:
        atts = list(range(len(codes[0])))
    nb_codes = len(codes)

    desc, targ, miss = [], [], []

    for c_idx in range(nb_codes):
        c_desc, c_targ, c_miss = code_to_query(codes[c_idx], atts)

        desc.append(c_desc)
        targ.append(c_targ)
        miss.append(c_miss)

    return desc, targ, miss


def code_to_query(code, atts=None):
    """
    Change the code-array to an actual queries, which are three arrays.

    :param code:                Array that contains:
                                     0 for desc attribute
                                     1 for target attribute
                                    -1 for missing attribute
    :param atts:                Array that contains the attributes (indices)
    :return: Three arrays.      One for desc atts indices, one for targets,
                                one for missing
    """

    if atts is None:
        atts = list(range(len(code)))
    assert len(code) == len(atts)

    desc = [x for i, x in enumerate(atts)
            if code[i] == encode_attribute(x,[x],[])]
    targ = [x for i, x in enumerate(atts)
            if code[i] == encode_attribute(x,[],[x])]
    miss = [x for i, x in enumerate(atts)
            if code[i] == encode_attribute(x, [], [])]
    return desc, targ, miss


def query_to_code(q_desc, q_targ, q_miss, atts=None):
    if atts is None:
        atts = determine_atts(q_desc, q_targ, q_miss)

    code = [encode_attribute(a, q_desc, q_targ) for a in atts]

    return code


def queries_to_codes(q_desc, q_targ, q_miss, atts=None):
    assert len(q_desc) == len(q_targ) == len(q_miss)
    nb_queries = len(q_desc)

    if atts is None:
        atts = determine_atts(q_desc[0], q_targ[0], q_miss[0])

    codes = [query_to_code(q_desc[i], q_targ[i], q_miss[i], atts=atts)
             for i in range(nb_queries)]

    return codes


def determine_atts(desc, targ, miss):
    """
    Determine the entire list of attributes.
    """
    atts = list(set(desc + targ + miss))
    atts.sort()
    return atts


def encode_attribute(att, desc, targ):
    """
    Encode the 'role' of an attribute in a model.

    `Role` means:
        - Descriptive attribute (input)
        - Target attribute (output)
        - Missing attribute (not relevant to the model)
    """

    check_desc = att in desc
    check_targ = att in targ

    code_int = check_targ * 2 + check_desc - 1

    return code_int


# Related to settings['metadata']
## Actual metadata
def get_metadata_df(df):
    """
    Get some useful statistics from a Pandas DataFrame.

    :param df:  Input DataFrame
    :return:    Dict with metadata
    #TODO(elia) is_nominal should become is_type
    """

    nb_tuples = df.shape[0]
    nb_atts = df.shape[1]

    # Initialize arrays (-1/False are no arbitrary choices!)
    nb_uvalues = np.full(nb_atts, -1)
    is_nominal = np.full(nb_atts, 0)
    tr_nominal = 20 # Our threshold of nb classes to call a column 'nominal'

    # Fill when necessary
    for i, col in enumerate(df.columns):
        if pd.api.types.is_integer_dtype(df[col]):
            nb_uvalues[i] = df[col].nunique()
            is_nominal[i] = 1 if nb_uvalues[i] < tr_nominal else 0

    metadata = {'is_nominal':   is_nominal,
                'nb_atts':      nb_atts,
                'nb_tuples':    nb_tuples,
                'nb_values':    nb_uvalues}

    return metadata


## Classlabels
def collect_and_verify_clf_classlabels(m_list, m_codes, is_nominal = None):
    """
    Collect the labels of the classifier.

    :param m_codes:
    :param m_list:
    :return:
    """

    _, m_targ, _ = codes_to_query(m_codes)

    nb_atts =  len(m_codes[0])
    clf_labels = [[0]] * nb_atts # Fill with dummy classes

    for m_idx, m in enumerate(m_list):
        # Collect the classlabels of one model
        nb_targ = len(m_targ[m_idx])
        m_classlabels = collect_classlabels(m, nb_targ, m_idx=m_idx)

        # Verify all the classlabels
        clf_labels = update_clf_labels(clf_labels, m_classlabels, m_targ[m_idx])

    return clf_labels


def collect_classlabels(m, nb_targ, m_idx=0):
    """
    Collect all the classlabels of a given model m.

    :param m:           Model
    :param nb_targ:     Number of target attributes
    :return:
    """
    # TODO: Fix the hotfix

    if hasattr(m, 'classes_'):
        if nb_targ == 1:
            # Single-target model
            if isinstance(m.classes_, list): # Hotfix.
                # This occurs when we ask classes_ of a model we built ourselves.
                m_classlabels = m.classes_
            else:
                m_classlabels = [m.classes_]

        else:
            # Multi-target model
            m_classlabels = [m.classes_[j] for j in range(nb_targ)]

    else:
        # If no classlabels are present, we assume a fully numerical model
        #warnings.warn("Model (ID: {}) has no classes_ attribute. Assuming only numerical targets".format(m_idx))
        m_classlabels  = [['numeric']] * nb_targ
        #TODO(elia): This needs to be done nicer.

    return m_classlabels


def fill_in_or_check_clf_labels(clf_labels, m_classlabels, m_targ):
    """
    Update (in case of default value) or check the consistency of the
    given classlabels of the model and its target attributes.

    :param clf_labels:
    :param m_classlabels:
    :param m_targ:
    :return:
    """

    for t_idx, t in enumerate(m_targ):
        if np.array_equal(clf_labels[t], [0]):
            # If the default value of [0] is still there
            clf_labels[t] = m_classlabels[t_idx]
        else:
            # Do a check whether what is provided is consistent
            assert np.array_equal(clf_labels[t], m_classlabels[t_idx])
    return clf_labels


def update_clf_labels(clf_labels, m_classlabels, m_targ):
    """
    Update the classlabels.

    Given an array of clf_labels, for each attribute known to the system,
    add the information on classlabels provided by a single model to it.
    This information is contained in m_classlabels, which are the classlabels
    known by the current model, on the targets as specified in m_targ.

    Update (in case of default value) or expand the present clf_labels.

    Clf_labels relates the MERCS system, and not to the individual
    classifiers.

    Parameters
    ----------
    clf_labels: list, shape (nb_atts, (nb_classlabels_att,))
        List of all classlabels known to the MERCS system
    m_classlabels: list, shape (nb_targ_atts_mod, (nb_classlabels_att,))
        List of all the classlabels known to the individual model
    m_targ
        List of all targets of the individual model

    Returns
    -------

    """
    # TODO: Deal with numeric in a smoother way

    for t_idx, t in enumerate(m_targ):

        old_labels = clf_labels[t]          # Classlabels already present in MERCS
        new_labels = m_classlabels[t_idx]   # Classlabels present in the model

        msg = "New_labels are: {}\n" \
              "Type new_labels is: {}\n".format(new_labels, type(new_labels))
        debug_print(msg, V=VERBOSITY, warn=True)

        msg = "Old_labels are: {}\n" \
              "Type old_labels is: {}\n".format(old_labels, type(old_labels))
        debug_print(msg, V=VERBOSITY, warn=True)

        assert isinstance(old_labels, (list, np.ndarray))
        assert isinstance(new_labels, (list, np.ndarray))

        if isinstance(old_labels, list):
            if old_labels == [0]:
                # If the default value of [0] is still there, update current
                clf_labels[t] = new_labels
            elif old_labels == ['numeric']:
                # If MERCS thought attribute t was numeric, the new model must agree!
                assert new_labels == ['numeric']
            else:
                msg = "type(old_labels) is list, but not the default value nor the default value for a numeric attribute.\n" \
                      "These are the only two allowed cases for an entry in clf_labels to be a list and not np.ndarray," \
                      "so something is wrong."
                raise TypeError(msg)
        elif isinstance(old_labels, np.ndarray):
            # Join current m_classlabels with those already present
            classlabels_list = [old_labels, new_labels]
            clf_labels[t] = join_classlabels(classlabels_list)
        else:
            msg = """
            old_labels (=clf_labels[t]) can only be a list or np.ndarray.\n
            A list can only occur in two case: \n
            \t 1) Default entry: [0] \n
            \t 2) Numeric dummy entry: ['numeric]\n\n
            """
            raise TypeError(msg)

    return clf_labels


def join_classlabels(classlabels_list):
    """
    Get the union of the provided classlabels

    This is crucial whenever models are trained on different subsets of the
    data_csv, and have other ideas about what the classlabels are.
    """

    all_classes = np.concatenate(classlabels_list)

    result = np.array(list(set(all_classes)))
    result.sort()

    return result


## Feature Importance
def collect_FI(m_list, m_codes):
    """
    Collect the feature importance of the models.

    :param m_codes:
    :param m_list:
    :return:
    """

    FI = np.zeros(m_codes.shape)
    m_desc, _, _ = codes_to_query(m_codes)

    for mod_i in range(len(m_list)):
        for desc_i, attr_i in enumerate(m_desc[mod_i]):
            FI[mod_i, attr_i] = m_list[mod_i].feature_importances_[desc_i]

    return FI
