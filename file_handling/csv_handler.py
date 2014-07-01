from collections import OrderedDict


def parse_csv_string(string):
    """Truncate a csv string into a list of lists.

    This method performs two operations:
      1. Split string into lines.
      2. Call parse_csv_list_of_strings
    """
    string_lines = string.split("\n")
    return parse_csv_list_of_strings(string_lines)


def parse_csv_list_of_strings(array):
    """Truncate a csv list of strings into a list of lists.

    This method performs the following operations:
      1. Check if one line is empty, if yes skip it
      2. Split lines, which are strings, at comas
      3. Remove unnecessary whitespace
    """
    csv_array = []
    for i in range(len(array)):
        if array[i]:
            fields = array[i].strip().split(",")
            for j in range(len(fields)):
                fields[j] = fields[j].strip()
            csv_array.append(fields)
    return csv_array


def convert_csv_array_to_ordered_dicts(csv_array, labels):
    ordered_dicts = list()
    for csv_list in csv_array:
        ordered_dicts.append(convert_csv_list_to_ordered_dict(csv_list,
                                                              labels))
    return ordered_dicts


def convert_csv_array_to_ordered_dicts_and_cast_types(csv_array, labels,
                                                      types):
    ordered_dicts = list()
    for csv_list in csv_array:
        ordered_dicts.append(convert_csv_list_to_ordered_dict_and_cast_types(
            csv_list, labels, types))
    return ordered_dicts


def convert_csv_list_to_ordered_dict(csv_list, labels):
    return convert_csv_list_to_ordered_dict_and_cast_types(csv_list, labels,
                                                         None)


def convert_csv_list_to_ordered_dict_and_cast_types(csv_list, labels, types):
    od = OrderedDict()
    if len(csv_list) != len(labels):
        raise ValueError("Length of CSV list (%d) is not equal to the number "
                         "of labels provided (%d)", (len(csv_list), len(labels)))
    for i, key_value in enumerate(zip(labels, csv_list)):
        key, value = key_value
        if key and value:
            if types is not None:
                value = types[i](value)
            od[key] = value
    return od
