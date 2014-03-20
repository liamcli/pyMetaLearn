__author__ = 'feurerm'

import cPickle
from collections import OrderedDict
import os
import time
import urllib2

import arff
import numpy as np
import pandas as pd

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.dataset import Dataset
from pyMetaLearn.metafeatures.metafeatures import calculate_all_metafeatures


class OpenMLDataset(Dataset):
    def __init__(self, source_community, id, name, version, description,
                 format, url, md5_checksum, local_directory, safe=True):
        super(OpenMLDataset, self).__init__(source_community, id, name,
                 version, description, format, url, md5_checksum,
                 local_directory, safe)

    def calculate_metadata(self):
        pass

    # todo: think whether the if/else is safe
    def get_processed_files(self):
        x_path = os.path.join(self.openMLBasePath(),
                              self._local_directory, "x.df")
        y_path = os.path.join(self.openMLBasePath(),
                              self._local_directory, "y.df")
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            self.get_unprocessed_files()
            x, y = self._prepare_dataset()
            x.to_pickle(x_path)
            y.to_pickle(y_path)
        else:
            with open(x_path) as x_pickle:
                x = cPickle.load(x_pickle)
            with open(y_path) as y_pickle:
                y = cPickle.load(y_pickle)
        return x, y

    def get_unprocessed_files(self):
        output_filename = "did" + str(self._id) + "_" + self._name + ".arff"
        output_path = os.path.join(self.openMLBasePath(),
                                   self._local_directory,
                                   output_filename)
        if not os.path.exists(output_path):
            return self._fetch_dataset(output_path)
        else:
            fh = open(output_path)
            arff_object = arff.load(fh)
            fh.close()
            return arff_object

    def _fetch_dataset(self, output_path):
        arff_string = self._read_url(self._url)
        arff_object = arff.loads(arff_string)
        fh = open(output_path, "w")
        fh.write(arff_string)
        fh.close()
        return arff_object

    def _prepare_dataset(self):
        starttime = time.time()
        data_frame = self._convert_arff_structure_to_pandas()

        class_ = data_frame.keys()[-1]
        attributes = data_frame.keys()[0:-1]
        x = data_frame[attributes]
        y = data_frame[class_]

        print "downloading dataset took", time.time() - starttime, "seconds."
        return x, y

    def _convert_arff_structure_to_pandas(self):
        data = self.get_unprocessed_files()
        data_dict = OrderedDict()
        for idx, attribute in enumerate(data["attributes"]):
            attribute_name = attribute[0].lower()
            attribute_type = attribute[1].lower() if type(attribute[1]) == \
                                                     str else "nominal"
            if attribute_type in set(['string, date']):
                raise NotImplementedError()

            # a string indicates something like real, integer while nominal
            # is represented as an array
            dtype = np.float64 if type(attribute[1]) == str else 'object'
            untransformed_array = [instance[idx] for instance in data["data"]]
            series = pd.Series(untransformed_array, name=attribute_name,
                               dtype=dtype)
            data_dict[attribute_name] = series

        assert pd.isnull(None)
        df = pd.DataFrame(data_dict, copy=True)
        del data_dict
        pd.set_option('display.max_columns', None)
        return df

    def render_as_html(self, file_handle):
        x, y = self.get_processed_files()
        html_table = x.to_html(float_format=lambda x: '%10f' % x,
            classes="table display", justify='right')
        file_handle.write(html_table)

    def render_as_csv(self, file_handle):
        x, y = self.get_processed_files()
        x.to_csv(file_handle)

    def get_metafeatures(self):
        metafeatures_filename = os.path.join(self.openMLBasePath(),
                                             self._local_directory,
                                             "metafeatures.pkl")
        if os.path.exists(metafeatures_filename):
            with open(metafeatures_filename) as fh:
                metafeatures = cPickle.load(fh)
        else:
            x, y = self.get_processed_files()
            metafeatures = calculate_all_metafeatures(x, y)
            with open(metafeatures_filename, "w") as fh:
                cPickle.dump(metafeatures, fh)
        return metafeatures

    def _read_url(self, url):
        connection = urllib2.urlopen(url)
        response = connection.read()
        return response

    def openMLBasePath(self):
        return pyMetaLearn.openml.manage_openml_data.get_local_directory()
