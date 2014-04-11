__author__ = 'feurerm'

import cPickle
from collections import OrderedDict
import os
import time
import urllib2

import arff
import numpy as np
import numpy.ma as ma
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

    def get_pandas(self):
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

    def get_npy(self):
        arff = self.get_unprocessed_files()
        x, y = self._convert_arff_structure_to_npy(arff)
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

    def _prepare_dataset(self, arff):
        starttime = time.time()
        data_frame = self._convert_arff_structure_to_pandas(arff)

        # TODO: the last attribute is not necessarily the class
        class_ = data_frame.keys()[-1]
        attributes = data_frame.keys()[0:-1]
        x = data_frame[attributes]
        y = data_frame[class_]

        print "downloading dataset took", time.time() - starttime, "seconds."
        return x, y

    def _convert_arff_structure_to_pandas(self, arff):
        """Has this interface to allow testing
        """
        data_dict = OrderedDict()
        for idx, attribute in enumerate(arff["attributes"]):
            attribute_name = attribute[0].lower()
            attribute_type = attribute[1].lower() if type(attribute[1]) == \
                                                     str else "nominal"
            if attribute_type in set(['string, date']):
                raise NotImplementedError()

            # a string indicates something like real, integer while nominal
            # is represented as an array
            dtype = np.float64 if type(attribute[1]) == str else 'object'
            untransformed_array = [instance[idx] for instance in arff["data"]]
            series = pd.Series(untransformed_array, name=attribute_name,
                               dtype=dtype)
            data_dict[attribute_name] = series

        assert pd.isnull(None)
        df = pd.DataFrame(data_dict, copy=True)
        del data_dict
        pd.set_option('display.max_columns', None)
        return df

    def _convert_arff_structure_to_npy(self, arff, replace_missing_with=0,
                                       scaling=None):
        """Nominal values are replaced with a one hot encoding and missing
         values represented with zero."""

        if replace_missing_with != 0:
            raise NotImplementedError()

        X, Y = self._prepare_dataset(arff)
        num_fields = 0
        attribute_arrays = []
        keys = []

        for idx, attribute in enumerate(X.iteritems()):
            attribute_name = attribute[0].lower()
            attribute_type = attribute[1].dtype
            row = attribute[1]

            if attribute_type == np.float64:
                rval = self._parse_numeric(row, scaling=scaling)
                if rval is not None:
                    keys.append(attribute_name)
                    attribute_arrays.append(rval)
                    num_fields += 1

            elif attribute_type == 'object':
                rval = self._parse_nominal(row)
                if rval is not None:
                    attribute_arrays.append(rval)
                    num_fields += rval.shape[1]
                    if rval.shape[1] == 1:
                        keys.append(attribute_name)
                    else:
                        vals = [attribute_name + ":" + str(possible_value) for
                                possible_value in range(rval.shape[1])]
                        keys.extend(vals)

            else:
                raise NotImplementedError()

        dataset_array = np.ndarray((X.shape[0], num_fields))
        col_idx = 0
        for attribute_array in attribute_arrays:
            length = attribute_array.shape[1]
            dataset_array[:, col_idx:col_idx + length] = attribute_array
            col_idx += length

        if Y.dtype == 'object':
            encoding = self.encode_labels(Y)
            Y = np.array([encoding[value] for value in Y], np.int32)
        elif Y.dtype == np.float64:
            Y = np.array([value for value in Y], dtype=np.float64)
        Y = Y.reshape((-1, 1))
        return dataset_array, Y


    def _parse_nominal(self, row):
        # This few lines perform a OneHotEncoding, where missing
        # values represented by none of the attributes being active (
        # a feature which i could not implement with sklearn).
        # Different imputation strategies can easily be added by
        # extracting a method from the else clause.
        # Caution: this methodology only keeps values that are
        # encountered in the dataset. If this is a subset of the
        # possible values of the arff file, only the subset is
        # encoded via the OneHotEncoding
        encoding = self.encode_labels(row)

        if len(encoding) == 0:
            return None

        array = np.zeros((row.shape[0], len(encoding)))

        for row_idx, value in enumerate(row):
            if row[row_idx] is not None:
                array[row_idx][encoding[row[row_idx]]] = 1

        return array


    def normalize_scaling(self, array):
        # Apply scaling here so that if we are setting missing values
        # to zero, they are still zero afterwards
        X_min = np.nanmin(array, axis=0)
        X_max = np.nanmax(array, axis=0)
        # Numerical stability...
        if (X_max - X_min) > 0.0000000001:
            array = (array - X_min) / (X_max - X_min)

        return array

    def normalize_standardize(self, array):
        raise NotImplementedError()
        mean = np.nanmean(array, axis=0, dtype=np.float64)
        X = array - mean
        std = np.nanstd(X, axis=0, dtype=np.float64)
        return X / std

    def _parse_numeric(self, row, scaling=None):
        # NaN and None will be treated as missing values
        array = np.array(row).reshape((-1, 1))

        if not np.any(np.isfinite(array)):
            raise NotImplementedError()

        if scaling == "normalize":
            array = self.normalize_scaling(array)
        elif scaling == "zero_mean":
            array = self.normalize_standardize(array)
        else:
            pass
        fixed_array = ma.fix_invalid(array, copy=True, fill_value=0)

        if not np.isfinite(fixed_array).all():
            print fixed_array
            raise NotImplementedError()

        return fixed_array


    def encode_labels(self, row):
        discrete_values = set(row)
        discrete_values.discard(None)
        discrete_values.discard(np.NaN)
        # Adds reproduceability over multiple systems
        discrete_values = sorted(discrete_values)
        encoding = OrderedDict()
        for row_idx, possible_value in enumerate(discrete_values):
            encoding[possible_value] = row_idx
        return encoding


    def render_as_html(self, file_handle):
        x, y = self.get_pandas()
        html_table = x.to_html(float_format=lambda x: '%10f' % x,
            classes="table display", justify='right')
        file_handle.write(html_table)

    def render_as_csv(self, file_handle):
        x, y = self.get_pandas()
        x.to_csv(file_handle)

    def get_metafeatures(self):
        metafeatures_filename = os.path.join(self.openMLBasePath(),
                                             self._local_directory,
                                             "metafeatures.pkl")
        if os.path.exists(metafeatures_filename):
            with open(metafeatures_filename) as fh:
                metafeatures = cPickle.load(fh)
        else:
            x, y = self.get_pandas()
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
