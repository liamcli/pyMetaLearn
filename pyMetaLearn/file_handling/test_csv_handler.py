import unittest
from collections import OrderedDict

import csv_handler


class TestCSVHandler(unittest.TestCase):
    def test_parse_csv_string(self):
        csv_string = " 2,4,6,7 , 6\n,Halle,Walle, Malle\nLa,La,La,La    ,La"
        array = csv_handler.parse_csv_string(csv_string)
        self.assertListEqual(array, [["2", "4", "6", "7", "6"],
                                     ["", "Halle", "Walle", "Malle",],
                                     ["La", "La", "La", "La", "La"]])

    def test_parse_csv_list_of_strings(self):
        csv_array = [" 2,4,6,7 , 6", ",Halle,Walle, Malle", "La,La,La,La  ,La"]
        array = csv_handler.parse_csv_list_of_strings(csv_array)
        self.assertListEqual(array, [["2", "4", "6", "7", "6"],
                                     ["", "Halle", "Walle", "Malle",],
                                     ["La", "La", "La", "La", "La"]])

    def test_parse_csv_list_of_strings_empty_lines(self):
        csv_array = [" 2,4,6,7 , 6", "", "La,La,La,La  ,La"]
        array = csv_handler.parse_csv_list_of_strings(csv_array)
        self.assertListEqual(array, [["2", "4", "6", "7", "6"],
                                     ["La", "La", "La", "La", "La"]])

    def test_convert_csv_array_to_ordered_dicts(self):
        csv_list = [["1", "2", "3", "4"], ["5", "6", "7", "8"]]
        labels = ["One", "Two", "Three", "Four"]
        ordered_dicts = csv_handler.convert_csv_array_to_ordered_dicts(
            csv_list, labels)
        self.assertIsInstance(ordered_dicts, list)
        self.assertEqual(len(ordered_dicts), 2)
        self.assertIsInstance(ordered_dicts[0], OrderedDict)

    def test_convert_csv_list_to_ordered_dict_fails(self):
        csv_list = ["2", "3", "4", "5"]
        labels = ["One", "Two", "Three"]
        self.assertRaises(ValueError,
                          csv_handler.convert_csv_list_to_ordered_dict,
                          csv_list, labels)

    def test_convert_csv_list_to_ordered_dict(self):
        csv_list = ["2", "3", "4", "5", ""]
        labels = ["One", "Two", "Three", "Four", ""]
        ret = csv_handler.convert_csv_list_to_ordered_dict(csv_list, labels)
        od = OrderedDict([("One", "2"), ("Two", "3"), ("Three", "4"),
                          ("Four", "5")])
        self.assertEqual(ret, od)