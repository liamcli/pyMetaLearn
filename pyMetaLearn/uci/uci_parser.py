from HTMLParser import HTMLParser
from bs4 import BeautifulSoup
import urllib2
import os
import numpy

class UCIDatasetOverviewParser(HTMLParser):
    """
    This parser extracts the datasets from the UCI website and returns a list of all dataset names.
    Example: http://archive.ics.uci.edu/ml/datasets.html?format=&task=cla&att=mix&area=&numAtt=&numIns=&type=&sort=nameUp&view=table

    A dataset has the following structure:

    <td><table><tr><td><a href="datasets/Abalone"><img src="assets/MLimages/SmallLarge1.jpg" border=1 /></a>&nbsp;</td><td><p class="normal"><b><a href="datasets/Abalone">Abalone</a></b></p></td></tr></table></td>
    <!-- <td><p class="normal">Predict the age of abalone from physical measurements&nbsp;</p></td> -->
    <td><p class="normal">Multivariate&nbsp;</p></td>
    <td><p class="normal">Classification&nbsp;</p></td>
    <td><p class="normal">Categorical, Integer, Real&nbsp;</p></td>
    <td><p class="normal">4177&nbsp;</p></td>
    <td><p class="normal">8&nbsp;</p></td>
    <td><p class="normal">1995&nbsp;</p></td>
    <!-- <td><p class="normal">Life&nbsp;</p></td> -->
    </tr><tr bgcolor="DDEEFF">
    """
    def __init__(self, datasets = None, uci_base_url = None, *args, **kwargs):
        HTMLParser.__init__(self, *args, **kwargs)
        self.datasets = datasets
        self.uci_base_url = uci_base_url

    def handle_starttag(self, tag, attrs):
        if tag == "a" and attrs[0][0] == "href" and "dataset" in attrs[0][1] and not "datasets.html" in attrs[0][1]:
            self.datasets[attrs[0][1][9:]] = self.uci_base_url + attrs[0][1][9:]


def fetch_uci_dataset_names(task=None, attribute_type=None, data_type=None, area=None, num_attributes=None,
                            num_instances=None, format_type=None):
    """
    Returns a dictionary with key being the dataset name and value the url of the dataset website.
    Check the options on the UCI website for possible options.
    """
    uci_base_url = "http://archive.ics.uci.edu/ml/datasets/"
    datasets = dict()
    parser = UCIDatasetOverviewParser(datasets=datasets, uci_base_url=uci_base_url)
    response = urllib2.urlopen("http://archive.ics.uci.edu/ml/datasets.html?format=%s&task=%s&att=%s&area=%s&numAtt=%s&numIns=%s&type=%s&sort=nameUp&view=table" %
                               (format_type, task, attribute_type, area, num_attributes, num_instances, data_type))
    html = response.read()
    parser.feed(html)
    return datasets


def fetch_website(link):
    try:
        response = urllib2.urlopen(link)
    except Exception as e:
        print "WARNING, link %s not found" % link
        print e
        return None
    html = response.read()
    return html


def fetch_dataset_metadata(html):
    """
    Retrieves the following meta-data from a dataset url:
    :Data Set Characteristics
    :Attribute Characteristics
    :Associated Tasks
    :Number of Instances
    :Number of Attributes
    :Missing Values?
    :Area
    :Date Donated
    :Data Set Information
    :Attribute Information
    :Data folder url
    :File links
    :Data folder name
    """
    uci_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    metadata = dict()
    meta_strings = ["Data Set Characteristics", "Number of Instances", "Area", "Attribute Characteristics",
                    "Number of Attributes", "Date Donated", "Associated Tasks", "Missing Values", "Data Set Information",
                    "Attribute Information"]

    dom_tree = BeautifulSoup(html)

    for link in dom_tree.find_all("a"):
        if link.get("href") is not None and "../machine-learning-databases" in link["href"] and\
                        "Data Folder" in link.string:
            metadata["data_folder_url"] = uci_dataset_url + link["href"][30:]
            metadata["data_folder_name"] = link["href"][30:-1]
    for paragraph in dom_tree.find_all("p"):
        if paragraph.get("class") is not None and\
                any([u"normal" == p_class for p_class in paragraph.get("class")]) and \
                paragraph.string is not None and\
                any([meta_string in paragraph.string for meta_string in meta_strings]):
            meta_string_found_idx = numpy.argmax([meta_string in paragraph.string for meta_string in meta_strings])
            meta_string = meta_strings[meta_string_found_idx]
            # For some reason u'\n' is also a sibling
            metadata[meta_string] = paragraph.parent.next_sibling.next_sibling.string

        if paragraph.get("class") is not None and\
                any([u"small-heading" == p_class for p_class in paragraph.get("class")]) and\
                paragraph.string is not None and\
                any([meta_string in paragraph.string for meta_string in meta_strings]):
            meta_string_found_idx = numpy.argmax([meta_string in paragraph.string for meta_string in meta_strings])
            meta_string = meta_strings[meta_string_found_idx]
            metadata[meta_string] = " ".join([string for string in paragraph.next_sibling.next_sibling.strings])

    return metadata


def fetch_dataset_filenames(html, base_url):
    unwanted_links = ["Index", "/ml/machine-learning-databases/", "?C=M;O=A", "?C=D;O=A", "?C=S;O=A", "?C=N;O=D", ]
    file_links = set()
    dom_tree = BeautifulSoup(html)

    for link in dom_tree.find_all("a"):
        if link.get("href") is not None and not any([unwanted == link["href"] for unwanted in unwanted_links]):
            file_links.add(base_url + link["href"])

    return file_links


if __name__ == "__main__":
    datasets = fetch_uci_dataset_names()
    print len(datasets)
    with open("uci_datasets.csv", "w") as csv:
        csv.write("Dataset Name;Associated Tasks;Number of Instances;Number of Attributes;Data Set Characteristics;"
                  "Attribute Characteristis;Missing Values?;Area;Number of Files;Data Folder;Date Donated\n")
        for dataset in datasets:
            # Datasets which are known to not work:
            broken_datasets = set(["Prodigy"])
            if dataset in broken_datasets:
                continue
            html = fetch_website(datasets[dataset])
            if html is None:
                continue
            metadata = fetch_dataset_metadata(html)
            print dataset, metadata
            html = fetch_website(metadata["data_folder_url"])
            if html is None:
                continue
            file_links = fetch_dataset_filenames(html, "data_folder_url")
            csv.write(";".join([dataset,
                                metadata["Associated Tasks"],
                                metadata["Number of Instances"],
                                metadata["Number of Attributes"],
                                metadata["Data Set Characteristics"],
                                metadata["Attribute Characteristics"],
                                metadata["Missing Values"],
                                metadata["Area"],
                                str(len(file_links)),
                                metadata["data_folder_name"],
                                metadata["Date Donated"]]))
            csv.write("\n")
            csv.flush()

