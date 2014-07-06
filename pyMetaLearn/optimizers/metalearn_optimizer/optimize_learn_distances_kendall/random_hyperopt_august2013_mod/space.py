from hyperopt import hp
from hyperopt.pyll import scope

space = {'n_estimators': scope.int(hp.quniform('n_estimators', 1, 100, 1)),
         'max_features': hp.uniform('max_features', 0.1, 1),
         'min_samples_split': scope.int(hp.quniform('min_samples_split', 1, 10, 1)),
         'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
         'seed': scope.int(hp.quniform('seed', 1, 100, 1)),
         'n_jobs': 4}
