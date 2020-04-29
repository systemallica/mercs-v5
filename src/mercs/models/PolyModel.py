from ..algo.inference import *
from ..algo.prediction import recode_strat
from ..utils.classlabels import collect_classlabels

from ..utils.debug import debug_print
VERBOSITY = 0


# Classes
class PolyModel(object):

    def __init__(self, m_list, m_desc, m_targ, targ, metadata):
        """
        PolyModel is a composite model.

        There are two main ways of composing a PolyModel:
            a) Ensemble-model:  This is aggregation of component models.
                                Geometrically speaking, this is 'horizontal'
                                combination
            b) Chained-model:   This is sequential combination of component
                                models.
                                Geometrically speaking, this is

        Parameters
        ----------
        m_list
        m_desc
        m_targ
        targ
        metadata
        """

        con = (targ, m_targ, m_desc, m_list)
        msg = "This PolyModel has\n" \
              "targ: {}\n" \
              "m_targ: {}\n" \
              "m_desc: {}\n" \
              "m_list: {}\n".format(*con)
        debug_print(msg, level=2, V=VERBOSITY, warn=True)

        # Models
        self.m_list = m_list

        # Attributes
        assert len(self.m_list) == len(m_desc)
        assert len(self.m_list) == len(m_targ)
        if not set(targ) <= set(np.concatenate(m_targ)):
            contents = (targ, m_targ, m_desc, m_list)
            msg = """
            This PolyModel has:
            \t targ: {}\n
            \t m_targ: {}\n
            \t m_desc: {}\n
            \t m_list: {}\n\n
            And this is a problem since we need
            targ to be a subset of the union of all m_targ.
            """.format(*contents)
            raise ValueError(msg)

        self.m_desc = m_desc
        self.m_targ = m_targ
        self.targ = targ

        # is_nominal
        self.is_attr_nominal = metadata['is_nominal']
        self.is_targ_nominal = [self.is_attr_nominal[t] for t in self.targ]

        self.targ_att_nominal = [v for i, v in enumerate(self.targ) if self.is_targ_nominal[i]]
        self.targ_att_numeric = [v for i, v in enumerate(self.targ) if not self.is_targ_nominal[i]]
        assert len(self.targ) == (len(self.targ_att_nominal) + len(self.targ_att_numeric))

        # Labels
        self.attr_lab = metadata['clf_labels']                                  # Get each attr label
        self.targ_lab = [self.attr_lab[t] for t in self.targ]                   # Get targ attr labels
        assert len(self.targ_lab) == len(self.targ)

        # TODO(elia): This needs to be done better. ALL the classes count! (also numeric ones)
        self.classes_ = [v for i, v in enumerate(self.targ_lab)
                         if self.is_targ_nominal[i]]
        assert np.sum(self.is_targ_nominal) == len(self.classes_)
        # self.classes_ = self.targ_lab

        # Active models (nominal/numeric)
        self.mod_idx_nominal = [i for i, v in enumerate(self.m_targ)
                                if (set(v) & set(self.targ_att_nominal))]
        self.mod_idx_numeric = [i for i, v in enumerate(self.m_targ)
                                if (set(v) & set(self.targ_att_numeric))]

        self.atts = list(range(metadata['nb_atts']))

        self.n_outputs_ = len(targ)

        return

    # Helpers
    def _get_mod_desc_targ(self, act_mod_idx):
        """
        Trivial helper for common combo

        :param act_mod_idx:     Int, index of the model you want to extract
        :return:
        """

        mod = self.m_list[act_mod_idx]
        mod_desc = self.m_desc[act_mod_idx]
        mod_targ = self.m_targ[act_mod_idx]
        return mod, mod_desc, mod_targ

    def _init_res_nominal(self, nb_samples):
        """
        Initialize datastructure for probabilities of nominal outcomes

        Datastructure:
            res_prob:       1 x n   list
                            n = Number of nominal targets
            res_prob[i]     a X b   np.array
                            a = nb_instances
                            b = number of labels for nominal target i

        :param nb_samples:
        :return:
        """

        nb_labels = [len(lab) for lab in self.classes_]
        res_nominal = [init_predictions(nb_samples, b)
                       for b in nb_labels]

        return res_nominal

    def _init_res_numeric(self, nb_samples):
        """
        Initialize datastructure for numeric predictions

        Datastructure:
            res_numer:      1 x n   list
                            n = Number of numeric targets
            res_numer[i]    a X 1   np.array
                            a = nb_instances

        :param nb_samples:
        :return:
        """

        nb_targ_att_numeric = len(self.targ_att_numeric)
        res_numeric = [init_predictions(nb_samples, 1)
                       for i in range(nb_targ_att_numeric)]

        return res_numeric

    def _generate_introspection_str(self):
        # TODO(elia): Generate a pretty print of everything in this object.
        return NotImplementedError


class EnsembleModel(PolyModel):
    """
    A single-layer group of models.

    This means an ensemble model.

    It is useful because it can cope with nominal, numeric and mixtures.
    """

    def __init__(self, m_list, m_desc, m_targ, targ, metadata):
        super().__init__(m_list, m_desc, m_targ, targ, metadata)

        # MAS
        self.mas = np.ones(len(self.m_list), dtype=int)  # Default: use all models

        # AAS
        self.aas = np.zeros(len(self.attr_lab), dtype=int)  # Initialize
        self.aas[self.targ] = 1  # Single layer, predict all at once

        return

    def predict(self, X):

        # Init datastructure
        nb_samples = X.shape[0]
        nb_targets = len(self.targ)
        predictions = init_predictions(nb_samples, nb_targets)

        if len(self.targ_att_nominal) > 0:
            res_proba = self.predict_proba(X)
            Y_nominal = predict_values_from_proba(res_proba, self.classes_)
            predictions = self.assemble_predictions(predictions,
                                                    Y_nominal,
                                                    mode='nominal')

        if len(self.targ_att_numeric) > 0:
            res_numer, counts = self.predict_numer(X)
            Y_numeric = predict_values_from_numer(res_numer, counts)
            predictions = self.assemble_predictions(predictions,
                                                    Y_numeric,
                                                    mode='numeric')

        return predictions

    def predict_proba(self, X):
        """
        Predict the probabilities of all the nominal targets.

        So this method ONLY yields results for the nominal targets.
        """

        # Basic check
        assert len(self.targ_att_nominal) > 0  # Otherwise nothing to predict

        # Init datastructure
        res_proba = self._init_res_nominal(X.shape[0])

        # Get active attributes
        res_atts = self.targ_att_nominal  # Only nominal targets
        res_labs = self.classes_  # Labels of these attributes

        # Get active models
        act_mod_idx = self.mod_idx_nominal

        # Do actual prediction
        for m_idx in act_mod_idx:
            mod, mod_desc, mod_targ = self._get_mod_desc_targ(m_idx)

            msg = """
            Model under consideration: \n
            model:\t{}\n
            mod_desc:\t{}\n
            mod_targ:\t{}\n
            """.format(mod, mod_desc, mod_targ)
            debug_print(msg, V=VERBOSITY)

            # Filter the nominal targets
            mod_targ_nominal = [self.is_attr_nominal[v] for v in mod_targ]
            mod_targ = [v for i, v in enumerate(mod_targ) if mod_targ_nominal[i]]

            mod_labs = collect_classlabels(mod)  # Collect labels of this model

            contents = (mod_labs,
                        mod_targ_nominal,
                        mod,
                        mod_targ,
                        mod_desc,
                        m_idx)
            msg = """
            mod_labs as collected by collect_classlabels: {}\n
            mod_targ_nominal: {} \n
            mod: {} \n \n
            mod_targ: {} \n
            mod_desc: {} \n
            m_idx: {} \n
            """.format(*contents)
            debug_print(msg, V=VERBOSITY, warn=True)

            mod_labs = [v for i, v in enumerate(mod_labs) if mod_targ_nominal[i]]

            mod_prob = mod.predict_proba(X[:, mod_desc])  # Individual prediction

            shared_targets = set(mod_targ) & set(res_atts)

            for t in shared_targets:
                t_idx_res = res_atts.index(t)  # Index of current target attr in result
                t_idx_mod = mod_targ.index(t)  # Index of current target attr in  current model

                res_proba = merge_proba(res_proba,
                                        mod_prob,
                                        res_labs,
                                        mod_labs,
                                        t_idx_res,
                                        t_idx_mod)

        return res_proba

    def predict_numer(self, X):
        """
        Collect all the predictions for the numeric targets.

        This method ONLY yields results for the numeric targets.
        """

        # Basic check
        assert len(self.targ_att_numeric) > 0  # Otherwise nothing to predict

        # Init datastructure
        res_numeric = self._init_res_numeric(X.shape[0])
        counts = [0] * len(res_numeric)  # Count amount of predictions for a single target

        # Get active attributes
        targ_res = self.targ_att_numeric  # Only numeric targets

        # Get active models
        act_mod_idx = self.mod_idx_numeric

        # Do actual prediction
        for m_idx in act_mod_idx:
            mod, desc_mod, targ_mod = self._get_mod_desc_targ(m_idx)
            nb_targ = len(targ_mod)

            mod_pred = mod.predict(X[:, desc_mod])  # Individual prediction

            targ_share = set(targ_mod) & set(targ_res)

            for t in targ_share:
                t_idx_res = targ_res.index(t)  # Index of target t in result
                t_idx_mod = targ_mod.index(t)  # Index of target t in  current model

                res_numeric = merge_numer(res_numeric,
                                          mod_pred,
                                          t_idx_res,
                                          t_idx_mod)
                counts[t_idx_res] += 1

            del mod_pred

        return res_numeric, counts

    def assemble_predictions(self, predictions, Y, mode=None):

        if mode in {'nominal'}:
            for t_idx_mod, t in enumerate(self.targ_att_nominal):
                t_idx_res = self.targ.index(t)  # Index of current target attr in result
                predictions[:, [t_idx_res]] = Y[:, [t_idx_mod]]
        elif mode in {'numeric'}:
            for t_idx_mod, t in enumerate(self.targ_att_numeric):
                t_idx_res = self.targ.index(t)  # Index of current target attr in result
                predictions[:, [t_idx_res]] = Y[:, [t_idx_mod]]
        else:
            msg = """
            Did not recognize mode: {}\n
            Cannot assemble predictions
            """.format(mode)
            raise ValueError(msg)

        return predictions


class ChainedModel(PolyModel):
    """
    A multi-layer group of models.

    This means an chained model.
    """

    def __init__(self, m_list, m_desc, m_targ, targ, metadata):
        super().__init__(m_list, m_desc, m_targ, targ, metadata)

        # MAS (CHANGE HERE)
        mas = np.array(range(1, len(self.m_list) + 1), dtype=int)
        assert len(np.unique(mas)) == len(mas)  # One model per layer
        self.mas, _ = recode_strat(mas, [])
        self.max_step = np.max(self.mas)

        # AAS (CHANGE HERE)
        aas = np.zeros(len(self.attr_lab), dtype=int)  # Initialize
        for m_idx, step in enumerate(mas):
            aas[m_targ[m_idx]] = step
        self.aas = aas

        return

    def predict(self, X):

        step = 1
        while step <= self.max_step:
            # Get active attributes/models
            act_mod_idx, act_att_idx = self.get_act_mod_att(step)
            assert len(act_mod_idx) == 1

            # Get model, m_desc, m_targ
            mod, mod_desc, mod_targ = self._get_mod_desc_targ(act_mod_idx[0])

            # Prediction
            mod_pred = mod.predict(X[:, mod_desc])

            # Update X
            X = update_X(X, mod_pred, act_att_idx)

            # Loop technics
            del mod_pred
            step += 1

        return X[:, self.targ]

    def predict_proba(self, X):
        # Basic check
        assert len(self.targ_att_nominal) > 0  # Otherwise nothing to predict

        # Init datastructure
        res_atts = self.targ_att_nominal
        res_labs = self.classes_
        res_prob = self._init_res_nominal(X.shape[0])

        step = 1
        while step <= self.max_step:
            # Get active attributes/models
            act_mod_idx, act_att_idx = self.get_act_mod_att(step)
            assert len(act_mod_idx) == 1

            # Get model, m_desc, m_targ
            mod, mod_desc, mod_targ = self._get_mod_desc_targ(act_mod_idx[0])

            # Prediction
            mod_pred = mod.predict(X[:, mod_desc])

            # Extract Prob.
            shared_targets = set(mod_targ) & set(res_atts)

            if len(shared_targets) > 0:
                mod_labs = mod.classes_
                mod_prob = mod.predict_proba(X[:, mod_desc])
                assert len(mod_labs) == len(mod_prob)

                for t in shared_targets:
                    t_idx_res = res_atts.index(t)  # Index of current target attr in result
                    t_idx_mod = mod_targ.index(t)  # Index of current target attr in  current model

                    res_prob = merge_proba(res_prob, mod_prob, res_labs, mod_labs, t_idx_res, t_idx_mod)

            # Update X
            X = update_X(X, mod_pred, act_att_idx)

            # Loop technics
            del mod_pred
            step += 1

        return res_prob

    # CHANGE
    def get_act_mod_att(self, step):
        """
        Get all the active models and attributes for given step.

        This can easily be extracted from:
            - mas:  Model Activation Strategy
            - aas:  Attribute Activation Strategy

        By *all* I mean that all the models and attributes are returned
        regardless if they are nominal or numerical.

        :param step:    Step (int) in the mas and aas.
        :return:
        """

        # Collect all active models and attributes
        act_mod_idx = np.where(np.array(self.mas) == step)[0]
        act_att_idx = np.where(np.array(self.aas) == step)[0]

        return act_mod_idx, act_att_idx


# Methods
def build_ensemble_model(m_list, targ, metadata):
    """
    From m_list, build an ensemble model.

    :param m_list:      List of models
    :param targ:        Target attributes of the EnsembleModel
    :param metadata:    Metadata dictionary
    :return:
    """

    # TODO(elia): This still assumes all attributes as descriptive one for the ensemble
    m_desc = [m.atts for m in m_list]
    m_targ = [m.targ for m in m_list]

    assert set(targ) <= set(np.concatenate(m_targ))  # targ is subset of all possible targets?

    return EnsembleModel(m_list, m_desc, m_targ, targ, metadata)


def build_chained_model(m_list, m_desc, m_targ, targ, mas, aas, metadata):
    """
    From mas and aas, build a ChainedModel.

    :param m_list:      List of models
    :param m_desc:      List of desc atts
    :param m_targ:      List of targ atts
    :param targ:        Target attributes of the ChainedModel
    :param mas:         Model Activation Strategy
    :param aas:         Attribute Activation Strategy
    :param metadata:    Metadata dictionary
    :return:
    """
    max_step = np.max(mas)
    assert max_step == np.max(aas)

    e_list = [None for i in range(max_step)]
    e_desc = [None for i in range(max_step)]
    e_targ = [None for i in range(max_step)]

    step = 1
    while step <= max_step:
        act_mod_idx = np.where(np.array(mas) == step)[0].tolist()
        act_att_idx = np.where(np.array(aas) == step)[0].tolist()

        act_mod = [m_list[idx] for idx in act_mod_idx]
        act_desc = [m_desc[idx] for idx in act_mod_idx]
        act_targ = [m_targ[idx] for idx in act_mod_idx]

        e_idx = step - 1
        e_list[e_idx] = EnsembleModel(act_mod, act_desc, act_targ, act_att_idx, metadata)
        e_targ[e_idx] = act_att_idx
        e_desc[e_idx] = list(range(len(aas)))  # For now, just all attributes

        # Loop technics
        step += 1

    return ChainedModel(e_list, e_desc, e_targ, targ, metadata)
