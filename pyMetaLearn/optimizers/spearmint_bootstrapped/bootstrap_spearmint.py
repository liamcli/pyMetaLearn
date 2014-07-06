import ast
from collections import OrderedDict
import os
import sys

import numpy as np

import HPOlib.wrapping_util as wrapping_util
import pyMetaLearn.optimizers.metalearn_optimizer.metalearner as ml


def main():
    print os.getcwd()
    config = wrapping_util.load_experiment_config_file()
    context = ml.setup(None)
    hp_list = ml.metalearn_suggest_all(None, context)
    num_bootstrap_examples = config.getint("METALEARNING",
                                         "num_bootstrap_examples")

    path_to_spearmint = config.get("SPEARMINT", "path_to_optimizer")
    import spearmint.main as spearmint

    for i, params in enumerate(hp_list[:num_bootstrap_examples]):
        fixed_params = OrderedDict()
        # Hack to remove all trailing - from the params which are
        # accidently in the experiment pickle of the current HPOlib version
        for key in params:
            if key[0] == "-":
                fixed_params[key[1:]] = params[key]
            else:
                fixed_params[key] = params[key]

        hp_list[i] = fixed_params

    sys.stderr.write("Initialize spearmint with " + str(hp_list[:num_bootstrap_examples]))
    sys.stderr.write("\n")
    sys.stderr.flush()

    spearmint.main(args=sys.argv[1:], pre_eval=hp_list[:num_bootstrap_examples])


if __name__ == "__main__":
    main()
