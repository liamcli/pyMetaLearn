from collections import OrderedDict
import glob
import os
import re

import file_handling.csv_handler


def _glob_for_file(path, filename):
    instances = glob.glob(os.path.join(path, filename))
    if len(instances) != 1:
        raise ValueError("Found %d paths for %s" %
                         (len(instances), os.path.join(path, filename)))
    return instances[0]


def get_state_run_file_paths(path):
    od = OrderedDict()
    od["instances_path"] = _glob_for_file(path, "instances.txt")
    od["runs_and_results_path"] = _glob_for_file(path, "runs_and_results-it*"
                                                       ".csv")
    od["scenario_path"] = _glob_for_file(path, "scenario.txt")
    od["param_file_path"] = _glob_for_file(path, "param-file.txt")
    od["uniq_configurations_path"] = _glob_for_file(path,
                                               "uniq_configurations-it*" \
                                                  ".csv")
    od["paramstrings_path"] = _glob_for_file(path, "paramstrings-it*.txt")
    return od


def parse_instances():
    raise NotImplementedError()


def parse_paramfile():
    raise NotImplementedError()


def parse_scenario():
    raise NotImplementedError()


def _parse_conditional(line):
    # Conditional Lines consist of:
	# <name1><w*>|<name2><w+><op><w+>{<values>} #Comment
	# <name1> - dependent parameter name
	# <name2> - the independent parameter name
	# <op> - probably only in at this time.
	# <w*> - zero or more whitespace characters.
	# <w+> - one or more whitespace characters
	# <values> - values
	#
	# Note: <op> only supports in at this time. (and the public interface
	# only assumes in).
	# @param line
    has_comment = False
    comment = ""
    if "#" in line:
        comment_begins = line.find("#")
        line = line[:comment_begins]
        comment = line[comment_begins:]
        has_comment = True

    if line.count("|") != 1 or line.count("{") != 1 or line.count("}") != 1:
        raise ValueError("Malformed parameter line %s" % line)

    condition_pipe = line.find("|")
    name = line[:condition_pipe].strip()
    first_bracket = line.find("{")
    second_bracket = line.find("}")
    in_operation = line[condition_pipe+1:first_bracket].find(" in ")
    conditional_on = line[condition_pipe+1:condition_pipe+in_operation+1]\
        .strip()
    condition_values = line[first_bracket+1:second_bracket].split(",")
    condition_values = [value.strip() for value in condition_values]

    od = OrderedDict()
    od["name"] = name
    od["conditional_on"] = conditional_on
    od["condition_values"] = condition_values
    od["has_comment"] = has_comment
    od["comment"] = comment
    return od


def _parse_forbidden(line):
    raise NotImplementedError()


def _parse_continuous(line):
     # Continuous Lines consist of:
	 #
	 # <name><w*>[<minValue>,<maxValue>]<w*>[<default>]<*w><i?><l?>#Comment
	 # where:
	 # <name> - name of parameter.
	 # <minValue> - minimum Value in Range
	 # <maxValue> - maximum Value in Range.
	 # <default> - default value enclosed in braces.
	 # <w*> - zero or more whitespace characters
	 # <i?> - An optional i character that specifies whether or not only
	 # integral values are permitted
	 # <l?> - An optional l character that specifies if the domain should be
	 # considered logarithmic (for sampling purposes).
    has_comment = False
    comment = ""
    if "#" in line:
        comment_begins = line.find("#")
        line = line[:comment_begins]
        comment = line[comment_begins:]
        has_comment = True

    if line.count("[") != 2 or line.count("]") != 2:
        raise ValueError("Malformed parameter line %s" % line)

    first_bracket = line.find("[")
    second_bracket = line.find("]")
    domain_values = line[first_bracket+1:second_bracket]
    cont_values = domain_values.split(",")
    if len(cont_values) != 2:
        raise ValueError("Expected two parameter values (or one comma between"
                         " the first brackets) in %s" % line)
    name = line[:first_bracket].strip()
    min = float(cont_values[0].strip())
    max = float(cont_values[1].strip())

    third_bracket = line.find("[", second_bracket)
    fourth_bracket = line.find("]", third_bracket)
    default_value = float(line[third_bracket+1:fourth_bracket].strip())

    line_end = line[fourth_bracket+1:]
    integer = True if "i" in line_end else False
    line_end = line_end.replace("i", "", 1)
    logarithmic = True if "l" in line_end else False
    line_end = line_end.replace("l", "", 1)
    if len(line_end.strip()) != 0:
        raise ValueError("Unknown or duplicate modifier(s): %s in %s" % (
            line_end, line))

    if len(line[second_bracket+1:third_bracket].strip()) != 0:
        raise ValueError("Invalid Characters detected between domain and "
                         "default value: %s" % line[
                                               second_bracket:third_bracket])

    if integer:
        min = int(min)
        max = int(max)
        default_value = int(default_value)

    od = OrderedDict()
    od["name"] = name
    od["type"] = "int" if integer else "float"
    od["min"] = min
    od["max"] = max
    od["default"] = default_value
    od["integer"] = integer
    od["logarithmic"] = logarithmic
    od["has_comment"] = has_comment
    od["comment"] = comment
    return od


def _parse_categorical(line):
	 # Categorical Lines consist of:
	 #
	 # <name><w*>{<values>}<w*>[<default>]<*w>#Comment
	 # where:
	 # <name> - name of parameter.
	 # <values> - comma seperated list of values (i.e. a,b,c,d...,z)
	 # <default> - default value enclosed in braces.
	 # <w*> - zero or more whitespace characters
    has_comment = False
    comment = ""
    if "#" in line:
        comment_begins = line.find("#")
        line = line[:comment_begins]
        comment = line[comment_begins:]
        has_comment = True

    if line.count("[") != 1 or line.count("]") != 1 or line.count("{") != 1 \
        or line.count("}") != 1:
        raise ValueError("Malformed parameter line %s" % line)

    first_bracket = line.find("{")
    second_bracket = line.find("}")
    domain_values = line[first_bracket+1:second_bracket]
    cat_values = domain_values.split(",")
    if len(cat_values) < 1:
        raise ValueError("Expected at least one value in %s" % line)
    name = line[:first_bracket].strip()
    values = [value.strip() for value in cat_values]

    third_bracket = line.find("[")
    fourth_bracket = line.find("]")
    default_value = line[third_bracket+1:fourth_bracket].strip()

    if len(line[second_bracket+1:third_bracket].strip()) != 0:
        raise ValueError("Invalid Characters detected between domain and "
                         "default value: %s" % line[
                                               second_bracket:third_bracket])

    od = OrderedDict()
    od["name"] = name
    od["type"] = "categorical"
    od["values"] = values
    od["default"] = default_value
    od["has_comment"] = has_comment
    od["comment"] = comment
    return od


def read_parameter_names_and_types(param_string):
    params = OrderedDict()
    lines = param_string.split("\n")
    for line in lines:
        # Logic kind of copied from SMAC ACLIB version 2.06.01-dev,
        # but a little bit more restrictive
        # file: ca.ubc.cs.beta.aclib.configspace.ParamConfigurationSpace.java
        # line 497-546
        type = ""
        if not line.strip():
            continue
        elif line.count("|") == 1:
            pass
            #print "WARNING: Conditionality is not parsed yet."
            #od = _parse_conditional(line)
        elif line.strip()[0] == "{":
            od = _parse_forbidden(line)
        elif line.count("[") == 0:
            if line.strip() == "Conditionals:":
                continue
            elif line.strip() == "Forbidden:":
                continue
            else:
                raise ValueError("Cannot parse the following line: %s" % line)
        elif line.count("[") == 2:
            od = _parse_continuous(line)
        elif line.count("{") == 1 and line.count("}") == 1:
            od = _parse_categorical(line)
        else:
            raise ValueError("Cannot parse the following line %s" % line)
        params[od["name"]] = od
    return params


def parse_paramstrings(string):
    """A special csv parser for SMAC paramstring files.

    These files contain lines in the following style:
    run_history_configuration_id: param1=value, param2=value, param3=value

    The format of the paramstrings document is not very well defined,
    therefore special cases are not implemented.
    """
    strings = string.split("\n")
    params = []
    for i, line in enumerate(strings):
        if line:
            p = OrderedDict()
            match = re.search(r"^([0-9]+:)", line)
            if match:
                _id = match.group(1)[:]  # The regex contains a colon
                param_string = line[len(_id):].strip()
                if param_string.count("=") * 2 != param_string.count("'"):
                    raise NotImplementedError("The case that there are not"
                                         " exactly one '=' and two ' per "
                                         "key/value pair is not implemented. "
                                         "We assume that the number of ' is "
                                         "twice the number of =, but it is %d "
                                         "and %d: %s" %
                                              (param_string.count("="),
                                               param_string.count("'"),
                                               param_string))
                for param in param_string.split(","):
                    key = ""
                    value = ""
                    if not param.strip():
                        raise ValueError("Found empty string in line %d" % (i
                                                                            + 1))
                    param = param.strip()
                    if param.count("=") != 1 and param.count("'") != 2:
                        raise NotImplementedError("The case that there is not"
                                         " exactly one '=' and two ' for a "
                                         "key/value pair is not implemented. "
                                         "This can happen if the value of a "
                                         "parameter contains a comma. "
                                         "# of =: %d, # of ': %d; "
                                         "%s" % (param.count("="),
                                               param.count("'"),
                                               param))
                    key_value = param.split("=")
                    key = key_value[0].strip()
                    value = key_value[1].strip().strip("'").strip()
                    if key and value:
                        p[key] = value
                    else:
                        raise ValueError("Found empty key or value. Key: %s; "
                                         "Value %s" % (key, value))
            else:
                raise ValueError("No Run History Configuration ID found in "
                                 "line %d" %i)
            params.append(p)
    return params


def parse_runs_and_results(string):
    """ A special csv parser for SMAC runs_and_results files.

    These files contain a line for every run in chronological order. The
    columns are as following:
      - Run number
      - Run history configuration ID: number of configuration
      - Instance ID: number of instance. Configuration ID and Instance ID
        form a unique identifier
      - Response Value (y)
      - Censored?
      - Cutoff time used
      - Runtime
      - Run length
      - Run result code
      - Run quality
      - SMAC iteration
      - SMAC cumulative runtime
      - Run result
      - Additional algorithm run data
      - Wall clock time
    """
    string_lines = string.split("\n")
    labels = "Run Number,Run History Configuration ID," \
             "Instance ID,Response Value (y),Censored?," \
             "Cutoff Time Used,Seed,Runtime,Run Length," \
             "Run Result Code,Run Quality,SMAC Iteration," \
             "SMAC Cumulative Runtime,Run Result," \
             "Additional Algorithm Run Data,Wall Clock Time,"
    assert string_lines[0] == labels
    labels = labels.split(",")
    types = [int, int, int, float, int, float, int, float, float, int, float,
             int, float, str, str, float]

    runs_and_results = file_handling.csv_handler.parse_csv_list_of_strings(
        string_lines[1:])
    runs = file_handling.csv_handler.convert_csv_array_to_ordered_dicts_and_cast_types(
        runs_and_results, labels, types)
    return runs


def parse_trajectory_string(string):
    """ A special csv parser for SMAC trajectory files.

    These files are normally called traj-run-i.csv, with i being a number. A
    trajectory file contains the following information:
      - Line 1 contains the rungroup and the seed
      - Line 2 contains the default configuration, the result is the worst
      possible one
      - From that line one, new lines are only added whenever the incumbent
      changes or improves.
    """
    string_lines = string.split("\n")
    trajectory_dict = OrderedDict()
    line0 = string_lines[0].split(",")
    if len(line0) != 2:
        raise ValueError("SMAC trajectory string contains not exactly two "
                         "elements on line one.")
    trajectory_dict["rungroup"] = line0[0]
    trajectory_dict["seed"] = int(float(line0[1]))
    trajectory = []
    trajectory_dict["trajectory"] = trajectory

    for line in string_lines[1:]:
        if line:
            split_line = line.split(", ")
            run = OrderedDict()
            run["start_time"] = float(split_line[0])
            run["mean_performance"] = float(split_line[1])
            run["incumbent_id"] = int(float(split_line[3]))
            run["params"] = OrderedDict()
            for key_value_pair in split_line[5:]:
                key, value = key_value_pair.split("=")
                value = value[1:-1]
                # Strip is necessary because there are two white spaces
                # before the parameters
                run["params"][key.strip()] = value
            trajectory.append(run)

    return trajectory_dict


