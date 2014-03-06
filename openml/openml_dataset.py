__author__ = 'feurerm'

from collections import OrderedDict
import os
import sys
import time
import urllib2

import arff
import numpy as np
import numpy.ma as ma
import pandas as pd

sys.path.append("..")
from dataset import Dataset


class OpenMLDataset(Dataset):
    def __init__(self, source_community, id, name, version, description,
                 format, url, md5_checksum, local_directory, safe=True):
        super(OpenMLDataset, self).__init__(source_community, id, name,
                 version, description, format, url, md5_checksum,
                 local_directory, safe)

    def calculate_metadata(self):
        pass

    def get_processed_files(self):
        X_path = os.path.join(self._output_directory, "X.npy")
        Y_path = os.path.join(self._output_directory, "Y.npy")
        if not (os.path.exists(X_path) and os.path.exists(Y_path)):
            data = self.get_unprocessed_files()
            X, Y = self._prepare_dataset(data)
            np.save(X_path, X)
            np.save(Y_path, Y)
        else:
            X = np.load(X_path)
            Y = np.load(Y_path)
        return X, Y

    def get_unprocessed_files(self):
        output_filename = "did" + str(self._id) + "_" + self._name + ".arff"
        output_path = os.path.join(self._output_directory, output_filename)
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

    def _prepare_dataset(self, data):
        starttime = time.time()
        dataset_array = self.convert_arff_structure_to_npy(data)

        X = dataset_array[:,:-1]
        Y = dataset_array[:,-1]

        print "Downloading dataset took", time.time() - starttime, "seconds."
        return X, Y

    def convert_arff_structure_to_pandas(self, data):
        data_dict = OrderedDict()
        for idx, attribute in enumerate(data["attributes"]):
            attribute_name = attribute[0].lower()
            attribute_type = attribute[1].lower() if type(attribute[1]) == \
                                                     str else "nominal"
            # A string indicates something like real, integer while nominal
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

    def convert_arff_structure_to_npy(self, data):
        """Nominal values are replaced with a one hot encoding and missing
         values represented with zero."""
        num_fields = 0
        attribute_arrays = []
        keys = []
        # Just a shortened version of how data["attributes" looks like
        # [(u'family', ['GB', 'GK', 'GS', 'TN', 'ZA', 'ZF', 'ZH', 'ZM', 'ZS']),
        # (u'enamelability', ['1', '2', '3', '4', '5']),
        # (u'bc', ['Y']), (u'bf', ['Y']),
        # (u'bt', ['Y']),
        # (u'bore', ['0', '500', '600', '760']),
        # (u'packing', ['1', '2', '3']),
        # (u'class', ['1', '2', '3', '4', '5', 'U'])]
        for idx, attribute in enumerate(data["attributes"]):
            attribute_name = attribute[0].lower()
            attribute_type = attribute[1].lower() if type(attribute[1]) == \
                                                     str else "nominal"

            if attribute_type in set(['string, date']):
                raise NotImplementedError()

            if attribute_type in set(['real', 'integer', 'numeric']):
                rval = self._parse_numeric(data, idx)
                if rval is not None:
                    keys.append(attribute_name)
                    attribute_arrays.append(rval)
                    num_fields += 1

            else:
                rval = self._parse_nominal(data, idx)
                if rval is not None:
                    attribute_arrays.append(rval)
                    num_fields += rval.shape[1]
                    if rval.shape[1] == 1:
                        keys.append(attribute_name)
                    else:
                        vals = [attribute_name + ":" + str(possible_value) for
                                possible_value in range(rval.shape[1])]
                        keys.extend(vals)

        dataset_array = np.ndarray((len(data["data"]), num_fields))
        col_idx = 0
        for attribute_array in attribute_arrays:
            length = attribute_array.shape[1]
            dataset_array[:, col_idx:col_idx + length] = attribute_array
            col_idx += length
        rs = np.random.RandomState(42)
        rs.shuffle(dataset_array)
        return dataset_array

    def _parse_nominal(self, data, idx):
        # This few lines perform a OneHotEncoding, where missing
        # values represented by none of the attributes being active (
        # a feature which i could not implement with sklearn).
        # Different imputation strategies can easily be added by
        # extracting a method from the else clause.
        # Caution: this methodology only keeps values that are
        # encountered in the dataset. If this is a subset of the
        # possible values of the arff file, only the subset is
        # encoded via the OneHotEncoding
        untransformed_array = [instance[idx] for instance in data[
            "data"]]
        discrete_values = set(untransformed_array)
        discrete_values.discard(None)
        # Adds reproduceability over multiple systems
        discrete_values = sorted(discrete_values)
        od = OrderedDict()
        for val_idx, possible_value in enumerate(discrete_values):
            od[possible_value] = val_idx

        # Target: only perform label encoding
        if idx == (len(data["attributes"]) - 1):
            array = np.array([od[instance[idx]] for instance in data[
                "data"]]).reshape((-1, 1))
            if not np.isfinite(array).all():
                raise NotImplementedError()

        else:
            if len(discrete_values) == 0:
                return None

            array = np.zeros(
                (len(untransformed_array), len(discrete_values)))

            val_idx = 0
            num_values = len(untransformed_array)
            while val_idx < num_values:
                while val_idx < num_values \
                        and untransformed_array[val_idx] is not None:
                    array[val_idx][od[untransformed_array[val_idx]]] = 1
                    val_idx += 1
                val_idx += 1

        return array

    def _parse_numeric(self, data, idx):
        # NaN and None will be treated as missing values
        array = np.array([instance[idx] if
                          (instance[idx] is not None and np.isfinite(
                              instance[idx]))
                          else np.NaN for instance in
                          data["data"]]).reshape((-1, 1))

        # Do not rescale the target classes...
        if idx != (len(data["attributes"]) - 1):
            # Check if there are any values available for that attribute,
            # otherwise kick out the attribute
            if not np.any(np.isfinite(array)):
                return None

            # Apply scaling here so that if we are setting missing values
            #  to zero, they are still zero afterwards
            X_min = np.nanmin(array, axis=0)
            X_max = np.nanmax(array, axis=0)
            # Numerical stability...
            if (X_max - X_min) > 0.0000000001:
                array = (array - X_min) / (X_max - X_min)

            # Replace invalid values (~np.isfinite)
            fixed_array = ma.fix_invalid(array, copy=True,
                                         fill_value=0)
        else:
            # By this it gets tested for finiteness
            fixed_array = array

        if not np.isfinite(fixed_array).all():
            print fixed_array
            raise NotImplementedError()

        return fixed_array

    def _read_url(self, url):
        connection = urllib2.urlopen(url)
        response = connection.read()
        return response
