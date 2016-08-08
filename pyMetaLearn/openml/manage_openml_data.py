import argparse
import ast
from collections import OrderedDict
import json
import os
import re
import StringIO
import tempfile
import urllib2
from urllib import urlencode

import xmltodict

try:
    from lxml import etree
except:
    pass

import pyMetaLearn.openml.openml_dataset
import pyMetaLearn.openml.openml_task


OPENML_DATA_DIR = os.path.abspath(
    os.path.expanduser(
    os.getenv("OPENML_DATA_DIR", os.path.join("~", ".OPENML_DATA_DIR"))))


def get_local_directory():
    if not os.path.isdir(OPENML_DATA_DIR):
        os.makedirs(OPENML_DATA_DIR)
    if not os.path.isdir(os.path.join(OPENML_DATA_DIR, "tasks")):
        os.mkdir(os.path.join(OPENML_DATA_DIR, "tasks"))
    if not os.path.isdir(os.path.join(OPENML_DATA_DIR, "metafeatures")):
        os.mkdir(os.path.join(OPENML_DATA_DIR, "metafeatures"))
    if not os.path.isdir(os.path.join(OPENML_DATA_DIR, "datasets")):
        os.mkdir(os.path.join(OPENML_DATA_DIR, "datasets"))
    return OPENML_DATA_DIR


def set_local_directory(newpath):
    global OPENML_DATA_DIR
    OPENML_DATA_DIR = newpath
    return get_local_directory()


def get_local_datasets():
    """Searches for all OpenML datasets in a given directory.

    Return a dictionary which maps dataset ids to the path of the xml file
    of the dataset"""
    directory = get_local_directory()
    dataset_directory = os.path.join(directory, "datasets")
    directory_content = os.listdir(dataset_directory)
    directory_content.sort()

    # Find all dataset ids for which we have downloaded the dataset description
    dataset_info = dict()
    for filename in directory_content:
        filepath = os.path.join(dataset_directory, filename)

        match = re.match(r"(did)([0-9]*)\.xml", filename)
        if match:
            did = match.group(2)
            did = int(did)

            dataset_info[did] = filepath

    datasets = OrderedDict()
    for did in sorted(dataset_info):
        datasets[did] = dataset_info[did]
    return datasets


def get_local_dataset(did):
    local_directory = get_local_directory()

    dataset_dir = os.path.join(local_directory, "datasets")
    dataset_file = os.path.join(dataset_dir, "did%d.xml" % int(did))
    with open(dataset_file) as fh:
        dataset = pyMetaLearn.openml.openml_dataset.OpenMLDataset\
            .from_xml_file(dataset_file)
    return dataset


def get_remote_datasets(names=False):
    """Return a list of all dataset ids. The list is obtained by parsing
    the response from  http://www.openml.org/api_query/free_query?q=SELECT+%60dataset%60.%60did%60+FROM+%60dataset%60
    """

    url = "http://openml.liacs.nl/api_query/free_query?q=" \
          "SELECT+%60dataset%60.%60did%60,%60dataset%60." \
          "%60name%60+FROM+%60dataset%60"
    json_string = _read_url(url)

    if names:
        return parse_dataset_id_name_json(json_string)
    else:
        return parse_dataset_id_json(json_string)


def get_remote_tasks():
    url = "http://openml.liacs.nl/api_query/free_query?q=SELECT+%60task%60" \
          ".%60task_id%60+FROM+%60task%60,%60task_type%60+" \
          "WHERE+%60task%60.%60ttid%60=%60task_type%60.%60ttid%60+" \
          "AND+%60task_type%60.%60name%60=%22Supervised%20Classification%22"
    json_string = _read_url(url)
    json_data = json.loads(json_string)

    assert "SQL was processed:" in json_data["status"]
    assert "Error" not in json_data["status"]

    task_ids = []
    for task in json_data["data"]:
        tid = ast.literal_eval(task[0])
        assert type(tid) is int
        task_ids.append(tid)
    return task_ids

def download_task(tid, cached=True):
    local_directory = get_local_directory()
    task_dir = os.path.join(local_directory, "tasks")
    xml_file = os.path.join(task_dir, "tid%d.xml" % tid)

    if not cached or not os.path.exists(xml_file):
        query_url = "http://api_new.openml.org/v1/task/%d"\
                    % tid
        try:
            task_xml = _read_url(query_url)
        except urllib2.URLError as e:
            print e, query_url
            raise e

        xml_file = xml_file
        with open(xml_file, "w") as fh:
            fh.write(task_xml)

    task = pyMetaLearn.openml.openml_task.OpenMLTask.from_xml_file(xml_file)

    print task
    return task


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


def download(dids, cached=True):
    """Downloads datasets.

    arguments:
    - dids: a single integer or a list of integers representing dataset ids.
    returns:
    - a list of dataset objects."""
    local_directory = get_local_directory()
    dataset_dir = os.path.join(local_directory, "datasets")
    datasets = []
    if isinstance(dids, int):
        dids = [dids]

    for did in dids:
        did = int(did)
        dataset_file = os.path.join(dataset_dir, "did%d.xml" % did)

        if not cached or not os.path.exists(dataset_file):
            if type(did) is not int:
                raise ValueError("%s is not of type integer. Are you sure that the "
                                 "argument dids %s is a list of integers?" % (did, dids))

            print "Fetching dataset id", did,
            query_url = "http://api_new.openml.org/v1/data/%d" % did

            try:
                dataset_xml = _read_url(query_url)
            except urllib2.URLError as e:
                print e, query_url
                raise e

            with open(dataset_file, "w") as fh:
                fh.write(dataset_xml)
            dataset = pyMetaLearn.openml.openml_dataset.OpenMLDataset\
                .from_xml_file(dataset_file)
            # Fetch the dataset from the internet
            dataset.get_unprocessed_files()

        dataset = pyMetaLearn.openml.openml_dataset.OpenMLDataset\
                .from_xml_file(dataset_file)
        datasets.append(dataset)
        print dataset._name
    return datasets

def _read_url(url):
    data={}
    data['api_key']='insert_priviate_key'
    data = urlencode(data)
    data = data.encode('utf-8')
    connection = urllib2.urlopen(url,data=data)
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    CHUNK = 16 * 1024
    string = StringIO.StringIO()
    with tmp as fh:
        while True:
            chunk = connection.read(CHUNK)
            if not chunk: break
            fh.write(chunk)

    tmp = open(tmp.name, "r")
    with tmp as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk: break
            string.write(chunk)

    return string.getvalue()


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


def list_local_datasets():
    """Print the local datasets from the method get_local_datasets."""
    directory = get_local_directory()
    dataset_info = get_local_datasets(directory)

    print " ID   - Name" + " " * 26
    for did in dataset_info:
        name = dataset_info[did]
        print "%5d: %-30s " % (did, name)


def show_remote_datasets():
    dids = get_remote_datasets(names=True)
    for did in dids:
        print did[0], did[1]


def show_only_remote():
    # TODO: this can be drastically sped-up by using an intersection-like
    # algorithm
    local_directory = get_local_directory()
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
