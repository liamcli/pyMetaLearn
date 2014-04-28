from collections import OrderedDict
import cPickle
import time

import numpy as np

import HPOlib.benchmark_util as benchmark_util
import HPOlib.wrapping_util as wrapping_util

def main(params, **kwargs):
    print 'Params: ', params
    folds = int(kwargs["folds"])
    fold = int(kwargs["fold"])

    config = wrapping_util.load_experiment_config_file()
    pickle_file = open(config.get("EXPERIMENT", "ground_truth"))
    results_pickle = cPickle.load(pickle_file)
    pickle_file.close()
    ground_truth = dict()
    measured_times = dict()

    for trial in results_pickle["trials"]:
        if not np.isfinite(trial["result"]) and \
            np.isfinite(trial["instance_results"]).all():
                raise ValueError("Results and instance_results must be valid "
                                 "numbers.")

        parameters = str(OrderedDict(sorted(trial["params"].items(),
                                        key=lambda t: t[0])))
        if folds > 1:
            ground_truth[parameters] = trial["instance_results"][fold]
            measured_times[parameters] = trial["instance_durations"][fold]
        else:
            ground_truth[parameters] = trial["result"]
            measured_times[parameters] = trial["duration"]

    params_hack = dict()
    for key in params:
        params_hack["-" + key] = params[key]
    params = str(OrderedDict(sorted(params_hack.items(), key=lambda t: t[0])))
    print ground_truth
    y = ground_truth[params]

    print 'Result: ', y
    return y

if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    # TODO return the time which was measured in the original pickle file!
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
