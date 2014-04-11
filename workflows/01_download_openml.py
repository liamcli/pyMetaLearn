import os
import pyMetaLearn.openml.manage_openml_data as data

# This should be the ids of the datasets used...
data.download(os.getenv("OPENML_DATA_DIR"),
              [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 21, 23, 24, 26, 27, 28,
               29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46,
               48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 113, 183,
               184, 185, 186, 188, 189, 190, 194, 2236])