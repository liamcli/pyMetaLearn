import os
import unittest
import shutil
import subprocess

import openml.manage_openml_data as manage_openml_data
import create_hpolib_dirs


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = os.path.abspath(".test_workflow_openml_to_libsvm")
        os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        pass

    def test_workflow(self):
        """Test the workflow of downloading a dataset from OpenML to running
        a SMAC experiment with the LibSVM through HPOlib on that dataset:
        1. Create a directory to perform experiments in (setUp method)
        2. Download a dataset from the openml server
        3. Retrieve the processed dataset and calculate meta-features
        4. Create an experiment directory
        5. Run a small gridsearch on the new experiment
        """

        datasets = manage_openml_data.get_remote_datasets(names=False)
        # This could either be a single dataset id or a list of dataset ids
        dataset = manage_openml_data.download(self.tmp_dir, datasets[0])
        # Dataset is actually a list of datasets
        self.assertEqual(len(dataset), 1)
        dataset = dataset[0]
        self.assertEqual(dataset._name, "anneal")

        # TODO: Y is always the last attribute, but in OpenML there is a flag
        #  for that...
        X, Y = dataset.get_pandas()

        # TODO: these do not yet add the class (Y) at the end...
        with open(os.path.join(self.tmp_dir, "anneal.html"), "w") as fh:
            # Pretty rendering with jQuery datatables
            fh.write('<style type="text/css" title="currentStyle">'
			         '@import "http://datatables.net/release-datatables/media/css/demo_page.css";'
			         '@import "http://datatables.net/release-datatables/media/css/demo_table.css";'
		             '</style>')
            fh.write('<script type="text/javascript" language="javascript" '
                     'src="http://datatables.net/release-datatables/media/js'
                     '/jquery.js"></script>\n')
            fh.write('<script type="text/javascript" language="javascript" '
                     'src="http://datatables.net/release-datatables/media/js/'
                     'jquery.dataTables.js"></script>\n')
            fh.write('<script type="text/javascript" charset="utf-8">'
			         '$(document).ready(function() {'
				     "$('.table').dataTable({'bPaginate': false});"
			         '} );'
		             '</script>')
            dataset.render_as_html(fh)

        with open(os.path.join(self.tmp_dir, "anneal.csv"), "w") as fh:
            dataset.render_as_csv(fh)

        # TODO: Calculate only those which are not already calculated
        mfs = dataset.get_metafeatures()
        self.assertEqual(len(mfs), 19)

        experiments_dir = os.path.join(self.tmp_dir, "experiments")
        os.mkdir(experiments_dir)

        ########################################################################
        # Create the template of an experiment directory...
        # TODO: is there a better way to generate these?
        experiment_dir = os.path.join(experiments_dir, "anneal_experiment")
        os.mkdir(experiment_dir)
        with open(os.path.join(experiment_dir, "config.cfg"), "w") as fh:
            content = create_hpolib_dirs.configure_config_template(
                "python -m pyMetaLearn.target_algorithm.libsvm",
                number_of_jobs=10)
            content += "\n[EXPERIMENT]"
            content += "\ntest_fold = 0"
            content += "\ntest_folds = 3"
            content += "\ndataset = OPENML:did1_anneal"
            content += "\ndataset_name = anneal"
            content += "\n"
            fh.write(content)
        with open(os.path.join(experiment_dir, "__init__.py"), "w") as fh:
            pass

        directories = dict()
        optimizers = ["gridsearch", "smac_2_06_01-dev",
                   "hyperopt_august2013_mod", "spearmint_april2013_mod"]
        for optimizer in optimizers:
            directories[optimizer] = os.path.join(experiment_dir, optimizer)
            os.mkdir(directories[optimizer])

        with open(os.path.join(directories["gridsearch"], "params.pcs"),
                  "w") as fh:
            fh.write("C {-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15}\n"
                     "gamma {-15, -13, -11, -9, -7, -5, -3, -1, 1, 3}")

        with open(os.path.join(directories["smac_2_06_01-dev"], "params.pcs"),
                  "w") as fh:
            fh.write("C [-5, 15] [1]\ngamma [-15, 3] [1]")

        with open(os.path.join(directories["hyperopt_august2013_mod"],
                  "space.py"), "w") as fh:
            fh.write("from hyperopt import hp\n\nspace = {'x': hp.uniform("
                     "'x', -5, 10),\n'y': hp.uniform('y', 0, 15)}\n")
        with open(os.path.join(directories["hyperopt_august2013_mod"],
                  "__init__.py"), "w") as fh:
            pass

        with open(os.path.join(directories["smac_2_06_01-dev"], "config.pb"),
                  "w") as fh:
            fh.write('language: PYTHON\nname:     "cv"\nvariable {\n' \
                     ' name: "x"\n type: FLOAT\n size: 1\n min:  -5\n' \
                     ' max:  10\n}\n\nvariable {\n name: "y"\n type: FLOAT\n' \
                     ' size: 1\n min:  0\n max:  15\n}')

        ########################################################################
        # Run the actual processes
        processes = []
        for fold_idx in range(3):
            # TODO:Better install it prior to using it to test whether this
            # works and to know where it resides
            cmd = "source ~/thesis/virtualenvs/pycharm/bin/activate\n"
            cmd += "export OPENML_DATA_DIR=~/thesis/datasets/openML/used/\n"
            cmd += "export PATH=$PATH:~/HPOlib/Software/runsolver_32\n"
            cmd += "export PATH=$PATH:~/HPOlib/Software/HPOlib/scripts\n"
            cmd += "export PYTHONPATH=$PYTHONPATH:~/HPOlib/Software/HPOlib\n"
            cmd += "export " \
                   "PYTHONPATH=$PYTHONPATH:~/thesis/Software/pyMetaLearn\n"
            cmd += "HPOlib-run -o /home/feurerm/HPOlib/working_directory/hpolib/optimizers" \
                   "/smac -s 1000 --cwd %s --EXPERIMENT:test_fold %d"\
                  % (experiment_dir, fold_idx)
            print cmd
            processes.append(subprocess.Popen(cmd, shell=True,
                                              executable="/bin/bash",
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE))
            out, err = processes[-1].communicate()
            print out
            print err
            self.assertEqual(processes[-1].returncode, 0)


if __name__ == "__main__":
    unittest.main()