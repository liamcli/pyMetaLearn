__author__ = 'feurerm'

import cPickle
from collections import OrderedDict
import lockfile
import os
import time
import types

import arff
import numpy as np
import numpy.ma as ma
import pandas as pd

import pyMetaLearn.openml.manage_openml_data
from pyMetaLearn.dataset import Dataset
from pyMetaLearn.metafeatures.metafeatures import calculate_all_metafeatures


class OpenMLDataset(Dataset):
    def __init__(self, source_community, id, name, version, description,
                 format, url, md5_checksum, local_directory,
                 default_target_attribute, safe=False):
        super(OpenMLDataset, self).__init__(source_community, int(id), name,
                 version, description, format, url, md5_checksum,
                 local_directory, safe)

    @classmethod
    def from_xml_file(cls, xml_file):
        local_dir = pyMetaLearn.openml.manage_openml_data.get_local_directory()
        dataset_dir = os.path.join(local_dir, "datasets")
        with open(xml_file, "r") as fh:
            task_xml = fh.read()
        dic = pyMetaLearn.openml.manage_openml_data._xml_to_dict(task_xml)["oml:data_set_description"]

        return cls("OpenML", dic["oml:id"], dic["oml:name"], dic["oml:version"],
            dic["oml:description"], dic["oml:format"], dic["oml:url"],
            dic["oml:md5_checksum"], dataset_dir, None)

    def calculate_metadata(self):
        pass

    def get_pandas(self, target=None):
        # TODO: add target to the filename
        x_path = os.path.join(self.openMLBasePath(), "datasets", "did%d_x.df"
                                                                 % self._id)
        y_path = os.path.join(self.openMLBasePath(), "datasets", "did%d_y.df"
                                                                 % self._id)
        import sys
        sys.stdout.flush()
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            arff = self.get_unprocessed_files()
            x, y = self._prepare_dataset(arff, target=target)
            x.to_pickle(x_path)
            y.to_pickle(y_path)
        else:
            # TODO: what happens if the target changes...
            with open(x_path) as x_pickle:
                x = cPickle.load(x_pickle)
            with open(y_path) as y_pickle:
                y = cPickle.load(y_pickle)
        return x, y

    def get_npy(self, target=None, replace_missing_with=0, scaling=None):
        # TODO: add target to the filename
        x_path = os.path.join(self.openMLBasePath(), "datasets", "did%d_x.npy"
                                                                 % self._id)
        y_path = os.path.join(self.openMLBasePath(), "datasets", "did%d_y.npy"
                                                                 % self._id)
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            arff = self.get_unprocessed_files()
            x, y = self._convert_arff_structure_to_npy(arff, target=target,
                replace_missing_with=replace_missing_with, scaling=scaling)
            with open(x_path, "w") as fh:
                np.save(fh, x)
            with open(y_path, "w") as fh:
                np.save(fh, y)
        else:
            with open(x_path) as fh:
                x = np.load(fh)
            with open(y_path) as fh:
                y = np.load(fh)
        return x, y

    def get_arff_filename(self):
        output_filename = "did" + str(self._id) + "_" + self._name + ".arff"
        output_path = os.path.join(self.openMLBasePath(), "datasets", output_filename)
        return output_path

    def get_unprocessed_files(self):
        output_path = self.get_arff_filename()
        print output_path
        if not os.path.exists(output_path):
            print "Download file"
            self._fetch_dataset(output_path)

        # A random number after which we consider a file for too large on a
        # 32 bit system...currently 120mb (just a little bit more than covtype)
        import struct
        bits = ( 8 * struct.calcsize("P"))
        if bits != 64 and os.path.getsize(output_path) > 120000000:
            return NotImplementedError("File too big")

        fh = open(output_path)
        arff_object = arff.load(fh)
        fh.close()
        return arff_object

    def _fetch_dataset(self, output_path):
        arff_string = self._read_url(self._url)

        fh = open(output_path, "w")
        fh.write(arff_string)
        fh.close()
        del arff_string

    def _prepare_dataset(self, arff, target=None):
        starttime = time.time()
        data_frame = self._convert_arff_structure_to_pandas(arff)

        # TODO: the last attribute is not necessarily the class

        if target is not None:
            try:
                x = data_frame.loc[:,data_frame.keys() != target]
                y = data_frame[target]
            except KeyError as e:
                print data_frame.keys(), target
                import sys
                sys.stdout.flush()
                raise e
        else:
            class_ = data_frame.keys()[-1]
            y = data_frame[class_]
            attributes = data_frame.keys()[0:-1]
            x = data_frame[attributes]

        # print "downloading dataset took", time.time() - starttime, "seconds."
        return x, y

    def _convert_attribute_type(self, attribute_type):
        # Input looks like:
        # {'?','GB','GK','GS','TN','ZA','ZF','ZH','ZM','ZS'}
        # real
        # etc...

        if isinstance(attribute_type, types.StringTypes):
            attribute_type = attribute_type.lower()
        elif isinstance(attribute_type, list):
            attribute_type = "nominal"
        else:
            raise NotImplementedError()

        # a string indicates something like real, integer while nominal
        # is represented as an array
        if attribute_type in ("real", "integer", "numeric"):
            dtype = np.float64
        elif attribute_type == "nominal":
            dtype = 'object'
        else:
            print attribute_type
            import sys
            sys.stdout.flush()
            raise NotImplementedError()

        return dtype

    def _convert_arff_structure_to_pandas(self, arff):
        # @attribute 'family' {'?','GB','GK','GS','TN','ZA','ZF','ZH','ZM','ZS'}
        """Has this interface to allow testing
        """
        data_dict = OrderedDict()
        for idx, attribute in enumerate(arff["attributes"]):
            attribute_name = attribute[0].lower()
            dtype = self._convert_attribute_type(attribute[1])

            untransformed_array = [instance[idx] for instance in arff["data"]]
            series = pd.Series(untransformed_array, name=attribute_name,
                               dtype=dtype)
            # Convert the name to lower space
            data_dict[attribute_name.lower()] = series

        assert pd.isnull(None)
        df = pd.DataFrame(data_dict, copy=True)
        del data_dict
        pd.set_option('display.max_columns', None)
        return df

    def _convert_arff_structure_to_npy(self, arff, replace_missing_with=0,
                                       scaling=None, target=None):
        """Nominal values are replaced with a one hot encoding and missing
         values represented with zero."""

        if replace_missing_with != 0:
            raise NotImplementedError(replace_missing_with)

        X, Y = self._prepare_dataset(arff, target)
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
        try:
            dataset_array = np.ndarray((X.shape[0], num_fields))
        except ValueError as e:
            print arff["relation"], X.shape[0], num_fields
            import sys
            sys.stdout.flush()
            raise e

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
        Y = Y.reshape((-1, 1)).ravel()

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

        if scaling == "scale":
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


    def get_metafeatures(self, subset_indices=None):
        # Bad implementation of metafeature caching...
        if type(subset_indices) not in (None, tuple):
            raise NotImplementedError(str(type(subset_indices)))

        metafeatures_filename = os.path.join(self.openMLBasePath(),
                                             "metafeatures",
                                             "metafeatures_did_%d.pkl" %
                                             self._id)

        if os.path.exists(metafeatures_filename):
            with open(metafeatures_filename) as fh:
                metafeatures = cPickle.load(fh)
        else:
            metafeatures = dict()

        if self._name not in metafeatures:
            metafeatures[self._name] = dict()
        if subset_indices not in metafeatures[self._name]:
            metafeatures[self._name][subset_indices] = dict()

        new = calculate_all_metafeatures(self, dont_calculate=metafeatures
            [self._name][subset_indices], subset_indices=subset_indices)
        if len(new) > 0:
            metafeatures[self._name][subset_indices].update(new)

            lock = lockfile.FileLock(metafeatures_filename)
            with lock:
                with open(metafeatures_filename, "w") as fh:
                    cPickle.dump(metafeatures, fh)

        return metafeatures[self._name][subset_indices]


    def _read_url(self, url):
        return pyMetaLearn.openml.manage_openml_data._read_url(url)

    def openMLBasePath(self):
        return pyMetaLearn.openml.manage_openml_data.get_local_directory()
