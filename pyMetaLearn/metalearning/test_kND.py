from collections import OrderedDict as OD
import unittest

import numpy as np
import pandas as pd

from pyMetaLearn.metalearning.kND import KNearestDatasets
from pyMetaLearn.metalearning.meta_base import Run


class kNDTest(unittest.TestCase):
    def setUp(self):
        self.anneal = pd.Series({"number_of_instances": 898., "number_of_classes": 5.,
                            "number_of_features": 38.}, name="anneal")
        self.krvskp = pd.Series({"number_of_instances": 3196., "number_of_classes":
                            2., "number_of_features": 36.}, name="krvskp")
        self.labor = pd.Series({"number_of_instances": 57., "number_of_classes":
                           2., "number_of_features": 16.}, name="labor")
        self.runs = {'anneal': [Run({'x': 0}, 0.1), Run({'x': 1}, 0.5), Run({'x': 2}, 0.7)],
                'krvskp': [Run({'x': 0}, 0.5), Run({'x': 1}, 0.1), Run({'x': 2}, 0.7)],
                'labor': [Run({'x': 0}, 0.5), Run({'x': 1}, 0.7), Run({'x': 2}, 0.1)]}

    def test_fit_l1_distance(self):
        kND = KNearestDatasets()

        kND.fit(pd.DataFrame([self.anneal, self.krvskp, self.labor]), self.runs)
        self.assertEqual(kND.best_hyperparameters_per_dataset['anneal'][0].params,
                         {'x': 0})
        self.assertEqual(kND.best_hyperparameters_per_dataset['krvskp'][0].params,
                         {'x': 1})
        self.assertEqual(kND.best_hyperparameters_per_dataset['labor'][0].params,
                         {'x': 2})
        self.assertTrue((kND.metafeatures ==
                         pd.DataFrame([self.anneal, self.krvskp, self.labor])).all().all())

    def test_kNearestDatasets(self):
        kND = KNearestDatasets()
        kND.fit(pd.DataFrame([self.krvskp, self.labor]),
                {'krvskp': self.runs['krvskp'], 'labor': self.runs['labor']})
        neighbor = kND.kNearestDatasets(self.anneal, 1)
        self.assertEqual([('krvskp', 1.8229893712531495)], neighbor)
        neighbors = kND.kNearestDatasets(self.anneal, 2)
        self.assertEqual([('krvskp', 1.8229893712531495),
                          ('labor', 2.2679197196559415)], neighbors)
        neighbors = kND.kNearestDatasets(self.anneal, -1)
        self.assertEqual([('krvskp', 1.8229893712531495),
                          ('labor', 2.2679197196559415)], neighbors)

        self.assertRaises(ValueError, kND.kNearestDatasets, self.anneal, 0)
        self.assertRaises(ValueError, kND.kNearestDatasets, self.anneal, -2)

    def test_kBestSuggestions(self):
        kND = KNearestDatasets()
        kND.fit(pd.DataFrame([self.krvskp, self.labor]),
                {'krvskp': self.runs['krvskp'], 'labor': self.runs['labor']})
        neighbor = kND.kBestSuggestions(self.anneal, 1)
        self.assertEqual([('krvskp', 1.8229893712531495, {'x': 1})], neighbor)
        neighbors = kND.kBestSuggestions(self.anneal, 2)
        self.assertEqual([('krvskp', 1.8229893712531495, {'x': 1}),
                          ('labor', 2.2679197196559415, {'x': 2})], neighbors)
        neighbors = kND.kBestSuggestions(self.anneal, -1)
        self.assertEqual([('krvskp', 1.8229893712531495, {'x': 1}),
                          ('labor', 2.2679197196559415, {'x': 2})], neighbors)

        self.assertRaises(ValueError, kND.kBestSuggestions, self.anneal, 0)
        self.assertRaises(ValueError, kND.kBestSuggestions, self.anneal, -2)

    def test_calculate_distances_to(self):
        kND = KNearestDatasets()
        a = pd.Series(data=[0, 1, 2, 17], name="a", dtype=np.float64)
        kND.metafeatures = pd.DataFrame(data=[a])
        b = pd.Series(data=[1, 3, 2, 3], name="b", dtype=np.float64)
        ret = kND._calculate_distances_to(b)
        self.assertEqual(3, ret['a'])

    def test_scale(self):
        kND = KNearestDatasets()
        metafeatures = pd.DataFrame([self.anneal, self.krvskp])
        metafeatures, other = kND._scale(metafeatures, self.labor)
        from pandas.util.testing import assert_series_equal
        # Series.equal does not work properly with floats...
        assert_series_equal(metafeatures.ix[0],
                            pd.Series({"number_of_instances": 0.267919719656,
                                      "number_of_classes": 1,
                                      "number_of_features": 1}))

    def test_l1(self):
        kND = KNearestDatasets()
        a = np.array([0, 1, 2, 17], dtype=np.float64)
        b = np.array([1, 3, 2, 3])
        self.assertEqual(kND._l1(a, b), 17)
        self.assertEqual(kND._calculate_distance(a, b), 17)

    def test_calculate_distances_l1(self):
        kND = KNearestDatasets(distance='l1')
        kND.metafeatures = pd.DataFrame(data=[self.krvskp, self.labor])
        distances = kND._calculate_distances_to(self.anneal)
        self.assertAlmostEqual(distances["krvskp"], 1.82298937196)
        self.assertAlmostEqual(distances["labor"], 2.267919719655)

    def test_l2(self):
        kND = KNearestDatasets(distance='l2')
        a = np.array([0, 1, 2, 17], dtype=np.float64)
        b = np.array([1, 3, 2, 3])
        self.assertAlmostEqual(kND._l2(a, b), 14.177446879)
        self.assertAlmostEqual(kND._calculate_distance(a, b), 14.177446879)

    def test_random(self):
        kND = KNearestDatasets(distance='random', random_state=1)
        a = np.array([0, 1, 2, 17], dtype=np.float64)
        b = np.array([1, 3, 2, 3])
        random_numbers = [kND._random(a, b) for i in range(20)]
        self.assertEqual(len(np.unique(random_numbers)), 20)

    def test_learned(self):
        kND = KNearestDatasets(distance='learned')
        rf = kND.fit(pd.DataFrame([self.krvskp, self.labor]),
                {'krvskp': self.runs['krvskp'], 'labor': self.runs['labor']})

        self.assertEqual(kND._learned(self.anneal, self.krvskp), 2)
        self.assertEqual(kND._learned(self.anneal, self.labor), 2)

    def test_feature_selection(self):
        kND = KNearestDatasets(distance='mfs_l1',
                               distance_kwargs={'max_features': 1.0,
                                                'mode': 'select'})
        self.krvskp.name = 'kr-vs-kp'
        selection = kND.fit(pd.DataFrame([self.krvskp, self.labor, self.anneal]),
                {'kr-vs-kp': self.runs['krvskp'],
                 'labor': self.runs['labor'],
                 'anneal': self.runs['anneal']})
        self.assertEqual(1, selection.loc['number_of_classes'])
        self.assertEqual(1, selection.loc['number_of_features'])
        self.assertEqual(0, selection.loc['number_of_instances'])

    def test_feature_weighting(self):
        kND = KNearestDatasets(distance='mfs_l1',
                               distance_kwargs={'max_features': 1.0,
                                                'mode': 'weight'})
        self.krvskp.name = 'kr-vs-kp'
        selection = kND.fit(pd.DataFrame([self.krvskp, self.labor, self.anneal]),
                {'kr-vs-kp': self.runs['krvskp'],
                 'labor': self.runs['labor'],
                 'anneal': self.runs['anneal']})
        self.assertEqual(type(selection), pd.Series)
        self.assertEqual(len(selection), 3)
