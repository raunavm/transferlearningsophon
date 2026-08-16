"""MTX network config: E1's Sophon architecture, unchanged, plus one eval-cost fix.

WHY THIS FILE EXISTS
--------------------
`experiments/E1/ParT_sophon_arch_10c.py` defines no `get_train_fn` /
`get_evaluate_fn`, so weaver falls back to its stock `train_classification` /
`evaluate_classification` (`weaver/utils/nn/tools.py:656-669`). Stock validation
runs `eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']`
(`tools.py:174`) on EVERY epoch. Two of those three scale very differently in K:

  * `roc_auc_score` is `partial(sklearn.roc_auc_score, multi_class='ovo')`
    (`metrics.py:49`). sklearn's OvO scores each pair on only that pair's
    samples, so the total is O(K * N), not O(K^2 * N). Measured at
    N_val = 1,280,000: 5.60 s at K=17, 14.63 s at K=43 -- a 2.61x rise for a
    2.53x rise in K, i.e. linear. Extrapolates to ~53 s/epoch at K=162. CHEAP.
    It stays.

  * `roc_auc_score_matrix` is weaver's own `roc_auc_score_ovo`
    (`metrics.py:25-37`), which calls sklearn K(K-1)/2 times on the FULL
    N-length arrays, zeroing the other classes with `sample_weight`. Cost is
    O(K^2 * N). Measured per-pair at N_val = 1,280,000: ~0.047 s, giving

        K=17   136 pairs    0.2 min/epoch     0.2 h over 80 epochs
        K=43   903 pairs    0.8 min/epoch     1.1 h over 80 epochs
        K=162 13041 pairs  10.2 min/epoch    13.5 h over 80 epochs

    (laptop CPU; cluster CPU is slower). That is a ~96x asymmetry between the
    two ends of the ladder and ~8% of L162's wall-clock, spent on a number
    nothing reads.

Nothing reads it because the training path returns `total_correct / count`
(`tools.py:263`) -- validation accuracy is the whole of `valid_metric`, and
`roc_auc_score_matrix` is computed, logged, and dropped. `evaluate_metrics`
wraps each metric in try/except (`metrics.py:69-74`), so this was never a crash
risk, only wall-clock.

SO: drop `roc_auc_score_matrix` during TRAINING VALIDATION only, and restore the
full set at test time, where the metrics are actually consumed and where it runs
exactly once rather than 80 times.

The architecture and loss are imported from the E1 file rather than copied, so
there is exactly one definition of the model and this file cannot drift from it.
E1's file is not edited -- its runs are finished and reproducible.

Invoke with:  -o num_classes <K> -o fc_params [(512,0.1)]
"""
import importlib.util
import os

_E1_ARCH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "E1", "ParT_sophon_arch_10c.py")

_spec = importlib.util.spec_from_file_location("_mtx_e1_arch", _E1_ARCH)
_e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e1)

# Re-exported verbatim. Do not redefine either of these here.
get_model = _e1.get_model
get_loss = _e1.get_loss

# Validation drops the quadratic metric; test keeps everything.
_TRAIN_EVAL_METRICS = ['roc_auc_score', 'confusion_matrix']
_TEST_EVAL_METRICS = ['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']


def get_train_fn(data_config, **kwargs):
    """Weaver's stock training loop, unmodified.

    weaver resolves `get_train_fn` and `get_evaluate_fn` together and falls back
    to BOTH stock functions if EITHER raises AttributeError (`train.py:655-669`).
    Defining only `get_evaluate_fn` would therefore be silently ignored. This
    returns the exact function the fallback would have supplied, so the training
    step is bit-identical to E1's.
    """
    from weaver.utils.nn.tools import train_classification
    return train_classification


def get_evaluate_fn(data_config, **kwargs):
    """Stock evaluation, with eval_metrics chosen by `for_training`."""
    from weaver.utils.nn.tools import evaluate_classification

    def _evaluate(model, test_loader, dev, epoch, for_training=True, **kw):
        kw.setdefault('eval_metrics',
                      _TRAIN_EVAL_METRICS if for_training else _TEST_EVAL_METRICS)
        return evaluate_classification(
            model, test_loader, dev, epoch, for_training=for_training, **kw)

    return _evaluate
