import numpy as np
import scipy
import tree
import export
import sklearn.svm
import sklearn.utils
import sklearn.ensemble
import cPickle
import StringIO
import pydot

# Does not work because the criterion needs some extra information
#import pyximport
#pyximport.install()
#import correlation_criterion_

input_file = "/home/feurerm/thesis/experiments/2014_04_25_predict_distances" \
             "/testfold_0-3_spearman_rank_perm.pkl"
with open(input_file) as fh:
    X, Y, metafeatures = cPickle.load(fh)

loo_mae = []
loo_mse = []
loo_rho = []
print "Dataset Mae MSE Rho"
for idx in range(metafeatures.shape[0]):
    rs = sklearn.utils.check_random_state(42)
    model = tree.DecisionTreeRegressor()
    forest = sklearn.ensemble.RandomForestRegressor(max_features=0.23,
                                                n_jobs=-1, random_state=rs,
                                                n_estimators=50)
    forest.base_estimator = model
    # forest.max_depth = 10
    forest.criterion = "kendall"
    leave_out_dataset = metafeatures.index[idx]

    train = []
    valid = []
    for cross in X.index:
        if leave_out_dataset in cross:
            valid.append(cross)
        else:
            train.append(cross)

    X_train = X.loc[train].values
    Y_train = Y.loc[train].values
    X_valid = X.loc[valid].values
    Y_valid = Y.loc[valid].values

    forest.fit(X_train, Y_train)

    predictions = forest.predict(X_valid)
    #dot_data = StringIO.StringIO()
    #export.export_graphviz(model, out_file=dot_data)
    #graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #graph.write_pdf("/home/feurerm/thesis/Software/pyMetaLearn/optimizers"
    #                "/metalearn_optimizer/correlation_tree/trees/%s.pdf" %
    #                leave_out_dataset)

    rho = scipy.stats.kendalltau(Y_valid, predictions)[0]
    mae = sklearn.metrics.mean_absolute_error(predictions, Y_valid)
    mse = sklearn.metrics.mean_squared_error(predictions, Y_valid)

    print leave_out_dataset, mae, mse, rho

    loo_mae.append(mae)
    loo_mse.append(mse)
    loo_rho.append(rho)
    # print np.mean(loo_mae), np.mean(mse), np.mean(loo_rho)

mae = np.mean(loo_mae)
mae_std = np.std(loo_mae)
mse = np.mean(loo_mse)
mse_std = np.std(loo_mse)
rho = np.mean(loo_rho)
rho_std = np.std(loo_rho)

print "MAE", mae, mae_std
print "MSE", mse, mse_std
print "Mean tau", rho, rho_std