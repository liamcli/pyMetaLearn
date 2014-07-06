from unittest import TestCase

__author__ = 'feurerm'

import unittest
import os
#from lxml import etree

import manage_openml_data
import arff


class TestManageOpenMLData(unittest.TestCase):
    def setUp(self):
         os.chdir(os.path.dirname(os.path.abspath(__file__)))

         fh = open("test_openml_restapi_anneal.xml")
         xml_array = fh.readlines()
         self.xml_string = " ".join(xml_array)
         fh.close()

         #with open("./schemas/dataset.xsd", 'r') as f:
         #    self.schema = etree.XMLSchema(etree.XML(f.read()))

         self.cached = True

    def test_parse_dataset_id_json(self):
         openml_free_query_response_string = \
             '{"status": "SQL was processed: 1076 rows selected. ",' \
             '"id": false,' \
             '"name": false,"time": 0.0019757747650146,"columns": ' \
             '[{"title":"did","datatype":"undefined"}],"data": [[' \
             '"2201", "some"],["187", "random"],["2236", "downloaded"],' \
             '["2023", "dataset"], ["1014", "aye"]]}'

         dataset_ids = manage_openml_data.parse_dataset_id_json(
             openml_free_query_response_string)

         self.assertListEqual(dataset_ids, [187, 1014, 2023, 2201, 2236])

    def test_parse_dataset_id_name_json(self):
        openml_free_query_response_string = \
             '{"status": "SQL was processed: 1076 rows selected. ",' \
             '"id": false,' \
             '"name": false,"time": 0.0019757747650146,"columns": ' \
             '[{"title":"did","datatype":"undefined"}],"data": [[' \
             '"2201", "some"],["187", "random"],["2236", "downloaded"],' \
             '["2023", "dataset"], ["1014", "aye"]]}'

        dataset_ids = manage_openml_data.parse_dataset_id_name_json(
             openml_free_query_response_string)

        self.assertListEqual(dataset_ids, [(187, u'random'), (1014, u'aye'),
            (2023, u'dataset'), (2201, u'some'), (2236, u'downloaded')])

    def test_download_task(self):
        print manage_openml_data.download_task(1)

    @unittest.SkipTest
    def test_validate_lxml(self):
         #xmlparser = etree.XMLParser(schema=self.schema)
         #etree.fromstring(self.xml_string, xmlparser)
        pass

    @unittest.SkipTest
    def test_get_local_datasets(self):
         self.fail()

