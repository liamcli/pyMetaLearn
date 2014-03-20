import argparse
import ast
from collections import OrderedDict
import json
import cPickle
import os
import re
import urllib2

from lxml import etree
import xmltodict

import pyMetaLearn.openml.openml_dataset


OPENML_DATA_DIR = os.path.abspath(
    os.path.expanduser(
    os.getenv("OPENML_DATA_DIR", os.path.join("~", "OPENML_DATA_DIR"))))


def get_local_directory():
    if not os.path.isdir(OPENML_DATA_DIR):
        os.makedirs(OPENML_DATA_DIR)
    return OPENML_DATA_DIR


def set_local_directory(newpath):
    global OPENML_DATA_DIR
    OPENML_DATA_DIR = newpath
    return get_local_directory()


def get_local_datasets(directory):
    """Searches for all OpenML datasets in a given directory."""
    directory_content = os.listdir(directory)
    directory_content.sort()

    # Find all dataset ids for which we have downloaded the dataset description
    dataset_info = dict()
    for filename in directory_content:
        filepath = os.path.join(directory, filename)

        match = re.match(r"(did)([0-9]*)(_)", filename)
        if match:
            did = match.group(2)
            did = int(did)

            fh = open(os.path.join(filepath, filepath + ".pkl"))
            dataset = cPickle.load(fh)
            fh.close()

            dataset_info[did] = dataset._name
            continue

    datasets = OrderedDict()
    for did in sorted(dataset_info):
        datasets[did] = dataset_info[did]
    return datasets


def get_remote_datasets(names=False):
    """Return a list of all dataset ids. The list is obtained by parsing
    the response from  http://www.openml.org/api_query/free_query?q=SELECT+%60dataset%60.%60did%60+FROM+%60dataset%60
    """

    url = "http://www.openml.org/api_query/free_query?q=" \
          "SELECT+%60dataset%60.%60did%60,%60dataset%60." \
          "%60name%60+FROM+%60dataset%60"
    json_string = _read_url(url)

    if names:
        return parse_dataset_id_name_json(json_string)
    else:
        return parse_dataset_id_json(json_string)


def parse_dataset_id_json(json_string):
    """Parse the json_string from get_remote_datasets, but return only the
    did"""
    parsed_and_sorted = parse_dataset_id_name_json(json_string)
    return [datum[0] for datum in parsed_and_sorted]


def parse_dataset_id_name_json(json_string):
    """Parse the json string returned obtained in the method
    "get_remote_datasets. Return a list with tuples, containing (did, name)"""
    json_data = json.loads(json_string)

    assert "SQL was processed:" in json_data["status"]
    assert "Error" not in json_data["status"]

    dataset_ids = []
    for datum in json_data["data"]:
        did = ast.literal_eval(datum[0])    # did = dataset id
        assert type(did) is int
        name = datum[1]
        dataset_ids.append((did, name))

    dataset_ids.sort()
    return dataset_ids


def download(local_directory, dids):
    """Downloads datasets.

    arguments:
    - local_directory: local working directory where the dataset will be
        downloaded to
    - dids: a single integer or a list of integers representing dataset ids.
    returns:
    - a list of dataset objects."""
    datasets = []
    if isinstance(dids, int):
        dids = [dids]

    for did in dids:
        if type(did) is not int:
            raise ValueError("%s is not of type integer. Are you sure that the "
                             "argument dids %s is a list of integers?" % (did, dids))

        print "Fetching dataset id", did,
        query_url = "http://www.openml.org/api/?f=openml.data" \
                ".description&data_id=%d" % did

        try:
            dataset_xml = _read_url(query_url)
        except urllib2.URLError as e:
            print e, query_url
            raise e

        descr = _parse_dataset_description(dataset_xml, local_directory)
        # Fetch the dataset from the internet
        descr.get_unprocessed_files()
        datasets.append(descr)
        print descr._name
    return datasets


def _parse_dataset_description(dataset_xml, local_directory):
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "schemas", "dataset.xsd")
    dataset_xsd = _read_file(schema_path)
    dic = _xml_to_dict(dataset_xml, dataset_xsd)["oml:data_set_description"]
    dataset_object = pyMetaLearn.openml.openml_dataset.OpenMLDataset(
        "OpenML", dic["oml:id"], dic["oml:name"], dic["oml:version"],
        dic["oml:description"], dic["oml:format"], dic["oml:url"],
        dic["oml:md5_checksum"], local_directory)

    return dataset_object

def _read_url(url):
    connection = urllib2.urlopen(url)
    response = connection.read()
    return response

def _read_file(filename):
    string = ""
    with open(filename, "r") as f:
        string = f.read()
    return string

def _xml_to_dict(xml_string, schema_string=None):
    # _validate_xml_against_schema(xml_string, schema_string)
    xml_dict = xmltodict.parse(xml_string)
    return xml_dict

def _validate_xml_against_schema(xml_string, schema_string):
    """Starting point for this code from http://stackoverflow.com/questions/
    17819884/xml-xsd-feed-validation-against-a-schema"""
    if type(schema_string) == str:
        schema = etree.XML(schema_string)
    elif type(schema_string) == etree.XMLSchema:
        schema = schema_string
    else:
        schema = etree.XMLSchema(schema_string)
    xmlparser = etree.XMLParser(schema=schema)
    etree.fromstring(xml_string, xmlparser)


def list_local_datasets(directory):
    """Print the local datasets from the method get_local_datasets."""
    dataset_info = get_local_datasets(directory)

    print " ID   - Name" + " " * 26
    for did in dataset_info:
        name = dataset_info[did]
        print "%5d: %-30s " % (did, name)


def show_remote_datasets():
    dids = get_remote_datasets(names=True)
    for did in dids:
        print did[0], did[1]


def show_only_remote(local_directory):
    # TODO: this can be drastically sped-up by using an intersection-like
    # algorithm
    remote_datasets_list = get_remote_datasets(names=True)
    remote_datasets = OrderedDict()
    for dataset in remote_datasets_list:
        remote_datasets[dataset[0]] = dataset[1]
    local_datasets = get_local_datasets(local_directory)
    remote_dids = set([did for did in remote_datasets])
    local_dids = set([did for did in local_datasets])
    differences = remote_dids.difference(local_dids)
    differences = list(differences)
    differences.sort()
    for did in differences:
        print did, remote_datasets[did]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_directory", default=None)
    tasks = parser.add_mutually_exclusive_group(required=True)
    # Show local datasets
    # TODO: Add argument to show only complete datasets
    tasks.add_argument("--list", help="List all datasets which are "
                       "found.", action="store_true")
    # Show datasets which are only on the remote server
    # TODO: Add argument to ignore incomplete datasets
    tasks.add_argument("--only-remote", help="Show datasets which are only "
                       "found on the OpenML server but not local.",
                       action="store_true")
    # Show remote datasets
    tasks.add_argument("--remote", help="List all datasets which are found on "
                       "OpenML server.", action="store_true")
    # Download single dataset
    # TODO: Add argument to download a batch of datasets
    tasks.add_argument("-d", "--download", help="Download a dataset.",
                       nargs="+", type=int)
    # Download all missing datasets
    tasks.add_argument("--download-all", help="Download all datasets",
                       action="store_true")
    args = parser.parse_args()

    if args.local_directory is None:
        raise ValueError("Please specify a local working directory, either "
                         "via the environment variable OPENML_DATA_DIR or the "
                         "argument --local-directory.")

    return args


def main():
    args = parse_arguments()

    if args.local_directory:
        set_local_directory(args.local_directory)

    if args.list:
        list_local_datasets(get_local_directory())
    elif args.remote:
        show_remote_datasets()
    elif args.only_remote:
        show_only_remote(get_local_directory())
    elif args.download:
        download(get_local_directory(), args.download)
    elif args.download-all:
        raise NotImplementedError()
    else:
        raise NotImplementedError("There is no other tasks this program can "
                                  "do...")


if __name__ == "__main__":
    main()