__author__ = 'feurerm'

import cPickle
from collections import OrderedDict, defaultdict
import gzip
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
            #arff = self.get_unprocessed_files()
            #x, y = self._convert_arff_structure_to_npy(arff, target=target,
            #    replace_missing_with=replace_missing_with, scaling=scaling)
            x, y = self.get_pandas(target=target)
            x, y = self._convert_pandas_to_npy(x, y,
                                               replace_missing_with=replace_missing_with,
                                               scaling=scaling)

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
        if os.path.exists(output_path):
            pass
        elif os.path.exists(output_path + ".gz"):
            output_path += ".gz"
        elif not os.path.exists(output_path):
            print "Download file"
            output_path += ".gz"
            self._fetch_dataset(output_path)

        # A random number after which we consider a file for too large on a
        # 32 bit system...currently 120mb (just a little bit more than covtype)
        import struct
        bits = ( 8 * struct.calcsize("P"))
        if bits != 64 and os.path.getsize(output_path) > 120000000:
            return NotImplementedError("File too big")

        if output_path[-3:] == ".gz":
            with gzip.open(output_path) as fh:
                arff_object = arff.load(fh)
        else:
            with open(output_path) as fh:
                arff_object = arff.load(fh)
        return arff_object

    def _fetch_dataset(self, output_path):
        arff_string = self._read_url(self._url)
        if output_path[-3:] != ".gz":
            output_path += ".gz"

        with gzip.open(output_path, "w") as fh:
            fh.write(arff_string)
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

    def convert_arff_structure_to_npy(self, arff, replace_missing_with=0,
                                       scaling=None, target=None):
        """Nominal values are replaced with a one hot encoding and missing
         values represented with zero."""

        if replace_missing_with != 0:
            raise NotImplementedError(replace_missing_with)

        X, Y = self._prepare_dataset(arff, target)
        return self._convert_pandas_to_npy(X, Y)

    def _convert_pandas_to_npy(self, X, Y, replace_missing_with=0,
                                       scaling=None):
        """Nominal values are replaced with a one hot encoding and missing
         values represented with zero."""

        if replace_missing_with != 0:
            raise NotImplementedError(replace_missing_with)

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

    def get_metafeatures(self, split_file_name=None, return_times=None,
                         return_helper_functions=False):
        if return_helper_functions:
            raise NotImplementedError()

        # Want a file because this enforces that the splits are actually
        # saved somewhere
        # Bad implementation of metafeature caching...
        # TODO: add folds and repearts to the calculation routine!
        splits_per_fold = defaultdict(list)
        repeat = 0
        if split_file_name is not None:
            with open(split_file_name) as fh:
                split_file = arff.load(fh)
                for line in split_file['data']:
                    if line[2] != 0:
                        raise NotImplementedError("No repeats implemented so far")
                    if line[0] == "TRAIN":
                        splits_per_fold[line[3]].append(line[1])
            split_file_name = os.path.split(split_file_name)[1]
        else:
            splits_per_fold[0] = None

        metafeatures_filename = os.path.join(self.openMLBasePath(),
                "metafeatures", "metafeatures_did_%d_subset_%s.arff" %
                                (self._id, split_file_name))

        if os.path.exists(metafeatures_filename):
            with open(metafeatures_filename) as fh:
                metafeatures = arff.load(fh)
                metafeatures['data'].sort()
        else:
            metafeatures = dict()
            metafeatures['relation'] = "metafeaturs_did_%d_%s" % (self._id,
                                                                  self._name)
            metafeatures['attributes'] = [('name', 'STRING'),
                                          ('type', 'STRING'),
                                          ('fold', 'NUMERIC'),
                                          ('repeat', 'NUMERIC'),
                                          ('value', 'NUMERIC'),
                                          ('time', 'NUMERIC')]

            metafeatures['data'] = []
        for fold in splits_per_fold:
            calculated_metafeatures = [feature[0] for feature in metafeatures['data']
                                       if feature[2] == fold]

            new, times = calculate_all_metafeatures(self,
                                             dont_calculate=calculated_metafeatures,
                                             subset_indices=splits_per_fold[fold],
                                             return_times=True)
            if len(new) > 0:
                for key in new:
                    metafeatures['data'].append([key, 'METAFEATURE', fold, repeat,
                                                 new[key], times[key]])

                for key in times:
                    if key not in new:
                        metafeatures['data'].append([key, 'HELPER_FUNCTION', fold,
                                                     repeat, '?' , times[key]])

        lock = lockfile.FileLock(metafeatures_filename)
        with lock:
            # Replace None values with a ?
            for idx in range(len(metafeatures['data'])):
                if metafeatures['data'][idx][4] is None:
                    metafeatures['data'][idx][4] = '?'
            with open(metafeatures_filename, "w") as fh:
                arff.dump(metafeatures, fh)

        # TODO: adapt for folds!
        if len(splits_per_fold) == 1 and splits_per_fold.keys()[0] == 0:
            metafeatures_dict =  dict([(line[0], line[4]) for line
                in metafeatures['data'] if line[1] == 'METAFEATURE'])
            times_dict = dict([(line[0], line[5]) for line
                in metafeatures['data'] if line[1] == 'METAFEATURE'])
            if return_times:
                return metafeatures_dict, times_dict
            return metafeatures_dict
        else:
            metafeatures_by_fold = defaultdict(dict)
            times_by_fold = defaultdict(dict)
            for metafeature in metafeatures['data']:
                if metafeature[1] != 'METAFEATURE': continue
                metafeatures_by_fold[metafeature[2]][metafeature[0]] = metafeature[4]
            for metafeature in metafeatures['data']:
                if metafeature[1] != 'METAFEATURE': continue
                times_by_fold[metafeature[2]][metafeature[0]] = metafeature[5]
            if return_times:
                return metafeatures_by_fold, times_by_fold
            return metafeatures_by_fold

    def _read_url(self, url):
        return pyMetaLearn.openml.manage_openml_data._read_url(url)

    def openMLBasePath(self):
        return pyMetaLearn.openml.manage_openml_data.get_local_directory()
