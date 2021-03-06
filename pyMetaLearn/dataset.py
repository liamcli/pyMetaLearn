import cPickle
import os


class Dataset(object):
    def __init__(self, source_community, id, name, version, description,
                 format, url, md5_checksum, base_dir, safe=True):
        # Attributes received by querying the RESTful API
        self.source = source_community
        self._id = id
        self._name = name
        self._version = version
        self._description = description
        self._comment = None
        self._format = format
        self._url = url
        self._md5_cheksum = md5_checksum
        self._meta_features = None

        # TODO: does anyone need this anymore?
        if safe:
            print base_dir
            self._local_directory = os.path.join(base_dir, "did" + str(id) +
                                                            "_" + self._name)
            if not os.path.exists(self._local_directory):
                os.makedirs(self._local_directory)

            filename = os.path.join(self._local_directory, "did" + str(id) + "_"
                                                            +  name + ".pkl")
            if not os.path.exists(filename):
                fh = open((filename), "w")
                cPickle.dump(self, fh)
                fh.close()

        else:
            self._local_directory = "./"