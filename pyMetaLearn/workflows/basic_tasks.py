import os


def create_directories(base_directory):
    directories = {}
    directories["base_experiments"] = \
        os.path.join(base_directory, "base_experiments")
    directories["datasets"] = os.path.join(base_directory, "datasets")
    directories["metafeatures"] = os.path.join(base_directory, "metafeatures")
    directories["pyMetaLearn_metafeatures"] = os.path.join(base_directory,
                                                      "pyMetaLearn_metafeatures")
    directories["metalearning_experiments"] = os.path.join(base_directory,
                                                      "metalearning_experiments")
    directories["plots"] = os.path.join(base_directory, "plots")
    directories["scripts"] = os.path.join(base_directory, "scripts")
    directories["splits"] = os.path.join(base_directory, "splits")
    directories["tasks"] = os.path.join(base_directory, "tasks")
    directories["custom_tasks"] = os.path.join(base_directory, "custom_tasks")

    for directory in directories:
        os.mkdir(directory)
