import unittest
from collections import OrderedDict
import os

import smac_extracter

class TestSMACExtracter(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def test_get_state_run_file_paths(self):
        path = "state-run1"
        state_run = smac_extracter.get_state_run_file_paths(path)
        self.assertIsInstance(state_run, OrderedDict)
        self.assertEqual(len(state_run), 6)
        self.assertEqual(state_run["runs_and_results_path"],
                         "state-run1/runs_and_results-it4.csv")

    def test_read_parameter_names_and_types(self):
        with open("state-run1/param-file.txt") as fh:
            param_string = fh.read()
        params = smac_extracter.read_parameter_names_and_types(param_string)
        self.assertIsInstance(params, OrderedDict)
        self.assertEqual(len(params), 786)

    def test_parse_contiuous_line(self):
        float_line = "_0__wcfmp_00_L [0.1, 1.0] [0.3]"
        param = smac_extracter._parse_continuous(float_line)
        self.assertDictEqual(param, OrderedDict([["name", "_0__wcfmp_00_L"],
                                                 ["type", "float"],
                                                 ["min", 0.1], ["max", 1.0],
                                                 ["default", 0.3],
                                                 ["integer", False],
                                                 ["logarithmic", False],
                                                 ["has_comment", False],
                                                 ["comment", ""]]))

        float_log_line = "_0__wcfsgd_01_L [1.0E-5, 0.1] [0.01]l"
        param = smac_extracter._parse_continuous(float_log_line)
        self.assertDictEqual(param, OrderedDict([["name", "_0__wcfsgd_01_L"],
                                                 ["type", "float"],
                                                 ["min", 0.00001],
                                                 ["max", 0.1],
                                                 ["default", 0.01],
                                                 ["integer", False],
                                                 ["logarithmic", True],
                                                 ["has_comment", False],
                                                 ["comment", ""]]))

        int_line = "_0__wcfvp_00_INT_I [1, 10] [1]i"
        param = smac_extracter._parse_continuous(int_line)
        self.assertDictEqual(param, OrderedDict([["name", "_0__wcfvp_00_INT_I"],
                                                 ["type", "int"],
                                                 ["min", 1], ["max", 10],
                                                 ["default", 1],
                                                 ["integer", True],
                                                 ["logarithmic", False],
                                                 ["has_comment", False],
                                                 ["comment", ""]]))

        int_log_line = "_0__wcfvp_01_INT_M [5000, 50000] [10000]il"
        param = smac_extracter._parse_continuous(int_log_line)
        self.assertDictEqual(param, OrderedDict([["name", "_0__wcfvp_01_INT_M"],
                                                 ["type", "int"],
                                                 ["min", 5000], ["max", 50000],
                                                 ["default", 10000],
                                                 ["integer", True],
                                                 ["logarithmic", True],
                                                 ["has_comment", False],
                                                 ["comment", ""]]))


    def test_parse_forbidden_line(self):
        self.assertRaises(NotImplementedError, smac_extracter
            ._parse_forbidden, "")

    def test_parse_conditional(self):
        line = "_1_00_0_QUOTE_START_B | _HIDDEN_ensemble_depth in {0, 1, 2, 3, 4}"
        param = smac_extracter._parse_conditional(line)
        self.assertDictEqual(param, OrderedDict([["name",
                                                  "_1_00_0_QUOTE_START_B"],
                                                 ["conditional_on",
                                                  "_HIDDEN_ensemble_depth"],
                                                 ["condition_values",
                                                  ["0", "1", "2", "3", "4"]],
                                                 ["has_comment", False],
                                                 ["comment", ""]
                                                 ]))

    def test_parse_categorical(self):
        self.maxDiff = None
        line = "_0__wcbbn_01_Q {weka.classifiers.bayes.net.search.local.K2, " \
               "weka.classifiers.bayes.net.search.local.HillClimber, " \
               "weka.classifiers.bayes.net.search.local.LAGDHillClimber," \
               " weka.classifiers.bayes.net.search.local.SimulatedAnnealing, " \
               "weka.classifiers.bayes.net.search.local.TabuSearch," \
               " weka.classifiers.bayes.net.search.local.TAN} " \
               "[weka.classifiers.bayes.net.search.local.K2]"
        param = smac_extracter._parse_categorical(line)
        self.assertDictEqual(param, OrderedDict([["name", "_0__wcbbn_01_Q"],
            ["type", "categorical"],
            ["values", ["weka.classifiers.bayes.net.search.local.K2",
                       "weka.classifiers.bayes.net.search.local.HillClimber",
                       "weka.classifiers.bayes.net.search.local"
                       ".LAGDHillClimber",
                       "weka.classifiers.bayes.net.search.local"
                       ".SimulatedAnnealing",
                       "weka.classifiers.bayes.net.search.local.TabuSearch",
                       "weka.classifiers.bayes.net.search.local.TAN"]],
             ["default", "weka.classifiers.bayes.net.search.local.K2"],
             ["has_comment", False],
             ["comment", ""]]))

    def test_read_paramstring_malformed_string(self):
        # Commata in parameter value, this is unfortunately legal, but should
        # be catched...
        string = "1: abc='jab,jab'"
        self.assertRaisesRegexp(NotImplementedError, "The case that there is not exactly one '=' and two ' for a key/value pair is not implemented. This can happen if the value of a parameter contains a comma. # of =: 0, # of ': 1; jab'",
                                smac_extracter.parse_paramstrings, string)
        # More than one equal sign
        string = "1: abc='jab=jab'"
        self.assertRaisesRegexp(NotImplementedError, "The case that there are not exactly one '=' and two ' per key/value pair is not implemented. We assume that the number of ' is twice the number of =, but it is 2 and 2: abc='jab=jab'",
                                smac_extracter.parse_paramstrings, string)
        # Quotation Marks are unfortunately completely legal although they
        # impose a lot of trouble.
        # Empty string
        string = "1: abc=''"
        self.assertRaisesRegexp(ValueError, "Found empty key or value. Key: abc; Value ",
                                smac_extracter.parse_paramstrings, string)
        # Key or value are empty
        string = "1: abc='jabjab', , bcd='jacjac'"
        self.assertRaisesRegexp(ValueError, "Found empty string in line 1",
                                smac_extracter.parse_paramstrings, string)
        pass

    def test_read_paramstrings(self):
        with open("SMAC-CV10-Termination-Convex-paramstrings-it4.txt") as fh:
            string = fh.read()
        params = smac_extracter.parse_paramstrings(string)
        self.assertIsInstance(params, list)
        self.assertEqual(len(params), 9)
        for p in params:
            self.assertIsInstance(p, OrderedDict)
            self.assertEqual(len(p), 786)
        self.assertEqual(params[3]["_1_02_1__wcfsgd_03_N"], "REMOVED")
        self.assertEqual(params[8]["_1_02_1__wcfsmo_03_3_REG_IGNORE_QUOTE_START_K"],
                         "weka.classifiers.functions.supportVector.Puk")


    def test_read_trajectory_file(self):
        with open("SMAC-CV10-Termination-Convex-traj-run-1.csv") as fh:
            string = fh.read()
        trajectory_dict = smac_extracter.parse_trajectory_string(string)
        self.assertIsInstance(trajectory_dict, OrderedDict)
        self.assertEqual(trajectory_dict["rungroup"], "autoweka")
        self.assertEqual(trajectory_dict["seed"], 1)

        self.assertEqual(len(trajectory_dict["trajectory"]), 6)

        for run in trajectory_dict["trajectory"]:
            self.assertIsInstance(run, OrderedDict)

        # Test some values...
        self.assertAlmostEqual(trajectory_dict["trajectory"][5]["start_time"],
                               111221.57796909999)
        self.assertAlmostEqual(trajectory_dict["trajectory"][5]
                               ["mean_performance"], 42.71825366666667)
        self.assertEqual(trajectory_dict["trajectory"][5]["incumbent_id"],
                         5)
        self.assertEqual(trajectory_dict["trajectory"][5]["params"]
                         ["_0__wcbbn_00_D"], "REMOVED")
        self.assertEqual(trajectory_dict["trajectory"][5]["params"]
                         ["_0__wctrept_04_2_INT_L"], "12")

    def test_read_runs_and_results_file(self):
        with open("SMAC-CV10-Termination-Convex-runs_and_results-it4.csv") as fh:
            string = fh.read()
        runs = smac_extracter.parse_runs_and_results(string)
        self.assertIsInstance(runs, list)
        self.assertEqual(len(runs), 21)

        for run in runs:
            self.assertIsInstance(run, OrderedDict)
            self.assertEqual(len(run), 16)

        self.assertEqual(runs[9]["Run Number"], 10)
        self.assertEqual(runs[9]["Run History Configuration ID"], 5)
        self.assertEqual(runs[9]["Instance ID"], 9)
        self.assertEqual(runs[9]["Response Value (y)"], 41.607143)
        self.assertEqual(runs[9]["Censored?"], False)
        self.assertEqual(runs[9]["Cutoff Time Used"], 9000.0)
        self.assertEqual(runs[9]["Seed"], -1)
        self.assertEqual(runs[9]["Runtime"], 10329.341)
        self.assertEqual(runs[9]["Run Length"], 0.0)
        self.assertEqual(runs[9]["Run Result Code"], 1)
        self.assertEqual(runs[9]["Run Quality"], 41.607143)
        self.assertEqual(runs[9]["SMAC Iteration"], 2)
        self.assertEqual(runs[9]["SMAC Cumulative Runtime"], 20794.0199891)
        self.assertEqual(runs[9]["Run Result"], "SAT")
        self.assertEqual(runs[9]["Additional Algorithm Run Data"], "EXTRA "
                                                                    "100.0")
        self.assertEqual(runs[9]["Wall Clock Time"], 10356.739)

