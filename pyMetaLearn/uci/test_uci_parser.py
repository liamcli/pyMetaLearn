import unittest
import os

from uci import uci_parser





class Test_UCI_Parser(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def test_fetch_dataset_metadata(self):
        html = " ".join(open("abalone.html").readlines())
        data_folder = " ".join(open("abalone_data_folder.html").readlines())

        metadata = uci_parser.fetch_dataset_metadata(html)
        expected_metadata = {"data_folder_url": u"http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/",
                             "data_folder_name": u"abalone",
                             "Data Set Characteristics": u"Multivariate",
                             "Number of Instances": u"4177",
                             "Area": u"Life",
                             "Attribute Characteristics": u"Categorical, Integer, Real",
                             "Number of Attributes": u"8",
                             "Date Donated": u"1995-12-01",
                             "Associated Tasks": u"Classification",
                             "Missing Values": u"No",
                             "Data Set Information": u"Predicting the age of abalone from physical measurements. "
                                                     " The age of abalone is determined by cutting the shell through the"
                                                     " cone, staining it, and counting the number of rings through a"
                                                     " microscope -- a boring and time-consuming task.  Other"
                                                     " measurements, which are easier to obtain, are used to predict"
                                                     " the age.  Further information, such as weather patterns and"
                                                     " location (hence food availability) may be required to solve the "
                                                     "problem.\n  \n From the original data examples with missing values"
                                                     " were removed (the majority having the predicted value missing),"
                                                     " and the ranges of the continuous values have been scaled for use"
                                                     " with an ANN (by dividing by 200).",
                             "Attribute Information": u"Given is the attribute name, " \
                                                      "attribute type, the measurement unit and a brief description." \
                                                      "  The number of rings is the value to predict: either as a" \
                                                      " continuous value or as a classification problem.\n  \n " \
                                                      "Name / Data Type / Measurement Unit / Description\n" \
                                                      "  -----------------------------\n  " \
                                                      "Sex / nominal / -- / M, F, and I (infant)\n  " \
                                                      "Length / continuous / mm / Longest shell measurement\n  " \
                                                      "Diameter\t/ continuous / mm / perpendicular to length\n  " \
                                                      "Height / continuous / mm / with meat in shell\n  " \
                                                      "Whole weight / continuous / grams / whole abalone\n  " \
                                                      "Shucked weight / continuous\t / grams / weight of meat\n  " \
                                                      "Viscera weight / continuous / grams / gut weight (after bleeding)\n  " \
                                                      "Shell weight / continuous / grams / after being dried\n  " \
                                                      "Rings / integer / -- / +1.5 gives the age in years\n  \n " \
                                                      "The readme file contains attribute statistics."}
        self.assertDictEqual(metadata, expected_metadata)

    def test_fetch_dataset_metadata_2(self):
        html = " ".join(open("abalone.html").readlines())
        data_folder = " ".join(open("abalone_data_folder.html").readlines())

        metadata = uci_parser.fetch_dataset_metadata(html)
        file_links = uci_parser.fetch_dataset_filenames(data_folder, "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/")
        expected_file_links = set()
        expected_file_links.add(u"http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
        expected_file_links.add(u"http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names")
        self.assertSetEqual(file_links, expected_file_links)