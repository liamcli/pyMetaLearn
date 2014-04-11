from collections import OrderedDict
import StringIO
import numpy as np
import unittest

import pandas as pd

import metalearner


class MetaLearnerTest(unittest.TestCase):
    def test_get_experiments_list(self):
        experiments_list = StringIO.StringIO()
        experiments_list.write(
            "/home/feurerm/thesis/experiments/2014_03_12_metaexperiments"
            "/did1_annealfold0/gridsearch_1_2014-2-24--20-3-30-871675"
            "/gridserach.pkl /home/feurerm/thesis/experiments/2014_03_12_metaexperiments"
            "/did1_annealfold1/gridsearch_1_2014-2-24--20-3-40-293595"
            "/gridsearch.pkl\n")
        experiments_list.write("\n")
        experiments_list.write(
            "/home/feurerm/thesis/experiments/2014_03_12_metaexperiments"
            "/did1_annealfold2/gridsearch_1_2014-2-24--20-3-26-901017"
            "/gridsearch.pkl\n")
        experiments_list.seek(0)
        retval = metalearner.read_experiments_list(experiments_list)
        self.assertEqual(len(retval), 3)
        self.assertEqual(len(retval[0]), 2)
        self.assertEqual(len(retval[1]), 0)


    def test_read_experiment_pickle(self):
        with open("gridsearch.pkl") as fh:
            rundata = metalearner.read_experiment_pickle(fh)
        self.assertEqual(len(rundata), 399)
        self.assertTrue(all([isinstance(run, metalearner.Experiment) for run
                             in rundata]))

    def test_l1(self):
        a = np.array([0, 1, 2, 17], dtype=np.float64)
        b = np.array([1, 3, 2, 3])
        self.assertEqual(metalearner.l1(a, b), 17)

    def test_l2(self):
        a = np.array([0, 1, 2, 17], dtype=np.float64)
        b = np.array([1, 3, 2, 3])
        self.assertAlmostEqual(metalearner.l2(a, b), 14.177446879)

    def test_vector_space_model(self):
        a = pd.Series([14, 5, 0, 2], dtype=np.float64) # sum of squares = 225 = 15x15
        sos_a = np.sqrt(sum([value**2 for value in a]))
        a_norm = a / sos_a
        b = pd.Series([4, 2, 1, 2], dtype=np.float64)  # sum of square = 25 = 5x5
        sos_b = np.sqrt(sum([value**2 for value in b]))
        b_norm = b / sos_b
        dist = sum(a_norm * b_norm)
        self.assertAlmostEqual(dist, metalearner.vector_space_model(a, b))

    def test_rescale(self):
        anneal = pd.Series({"number_of_instances": 898., "number_of_classes": 5.,
                            "number_of_features": 38.}, name="anneal")
        krvskp = pd.Series({"number_of_instances": 3196., "number_of_classes":
                            2., "number_of_features": 36.}, name="kr-vs-kp")
        labor = pd.Series({"number_of_instances": 57., "number_of_classes":
                            2., "number_of_features": 16.}, name="labor")
        data = pd.DataFrame([anneal, krvskp, labor])
        data = metalearner.rescale(data)
        from pandas.util.testing import assert_series_equal
        # Series.equal does not work properly with floats...
        assert_series_equal(data.ix[0],
                            pd.Series({"number_of_instances": 0.267919719656,
                                      "number_of_classes": 1,
                                      "number_of_features": 1}))

    def test_calculate_distances_vsp(self):
        anneal = pd.Series({"number_of_instances": 898., "number_of_classes": 5.,
                            "number_of_features": 38.}, name="anneal")
        krvskp = pd.Series({"number_of_instances": 3196., "number_of_classes":
                            2., "number_of_features": 36.}, name="kr-vs-kp")
        labor = pd.Series({"number_of_instances": 57., "number_of_classes":
                            2., "number_of_features": 16.}, name="labor")
        distance_function = metalearner.vector_space_model
        distances = metalearner.calculate_distances(anneal,
                        pd.DataFrame([krvskp, labor]), distance_function)
        self.assertEqual(distances[0][1], "labor")
        self.assertAlmostEqual(distances[0][0], 0.97297137561942704)
        self.assertEqual(distances[1][1], "kr-vs-kp")
        self.assertAlmostEqual(distances[1][0], 0.99950650794593432)

    def test_calculate_distances_l1(self):
        anneal = pd.Series({"number_of_instances": 0.267920, "number_of_classes": 1., "number_of_features": 1.}, name="anneal")
        krvskp = pd.Series({"number_of_instances": 1., "number_of_classes": 0., "number_of_features": 0.909091}, name="kr-vs-kp")
        labor = pd.Series({"number_of_instances": 0., "number_of_classes": 0., "number_of_features": 0.}, name="labor")
        distance_function = metalearner.l1
        distances = metalearner.calculate_distances(anneal,
                        pd.DataFrame([krvskp, labor]), distance_function)
        self.assertEqual(distances[0][1], "kr-vs-kp")
        self.assertAlmostEqual(distances[0][0], 1.822989)
        self.assertEqual(distances[1][1], "labor")
        self.assertAlmostEqual(distances[1][0], 2.2679200000000002)

    def test_split_metafeature_array(self):
        anneal = pd.Series({"number_of_instances": 898., "number_of_classes": 5.,
                            "number_of_features": 38.}, name="anneal")
        krvskp = pd.Series({"number_of_instances": 3196., "number_of_classes":
                            2., "number_of_features": 36.}, name="kr-vs-kp")
        labor = pd.Series({"number_of_instances": 57., "number_of_classes":
                            2., "number_of_features": 16.}, name="labor")
        metafeatures = pd.DataFrame([anneal, krvskp, labor])
        ds_metafeatures, other_metafeatures = metalearner\
            .split_metafeature_array("kr-vs-kp", metafeatures)
        self.assertIsInstance(ds_metafeatures, pd.Series)
        self.assertEqual(len(other_metafeatures.index), 2)

    def test_select_params(self):
        best_hyperparameters = dict()
        best_hyperparameters["kr-vs-kp"] = OrderedDict([("C", 1)])
        best_hyperparameters["labor"] = OrderedDict([("C", 2)])
        distances = [(0.1, "kr-vs-kp"), (0.2, "labor")]
        exp1 = metalearner.Experiment(OrderedDict([("C", 1)]), 0.1)
        exp2 = metalearner.Experiment(OrderedDict([("C", 2)]), 0.2)
        params = metalearner.select_params(best_hyperparameters, distances, [])
        self.assertEqual(params, OrderedDict([("C", 1)]))
        params = metalearner.select_params(best_hyperparameters, distances, [exp1])
        print params
        self.assertEqual(params, OrderedDict([("C", 2)]))
        params = metalearner.select_params(best_hyperparameters, distances, [exp2])
        self.assertEqual(params, OrderedDict([("C", 1)]))
        self.assertRaises(StopIteration, metalearner.select_params,
            best_hyperparameters, distances, [exp1, exp2])

    def test_find_best_hyperparams(self):
        exp0 = metalearner.Experiment(OrderedDict([("C", 0)]), 0.2)
        exp1 = metalearner.Experiment(OrderedDict([("C", 1)]), 0.5)
        exp2 = metalearner.Experiment(OrderedDict([("C", 2)]), 0.3)
        exp3 = metalearner.Experiment(OrderedDict([("C", 3)]), 0.1)
        exp4 = metalearner.Experiment(OrderedDict([("C", 4)]), 0.15)
        exp5 = metalearner.Experiment(OrderedDict([("C", 5)]), np.NaN)
        exp6 = metalearner.Experiment(OrderedDict([("C", 6)]), np.Inf)
        self.assertEqual(metalearner.find_best_hyperparams(
            [exp0, exp1, exp2, exp3, exp4, exp5, exp6]), OrderedDict([("C", 3)]))
        self.assertEqual(metalearner.find_best_hyperparams(
            [exp6, exp5, exp4, exp3, exp2, exp1, exp0]), OrderedDict([("C", 3)]))


if __name__ == "__main__":
    unittest.main()

