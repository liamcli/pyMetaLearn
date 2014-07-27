##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
import sys

logger = logging.getLogger("HPOlib.optimizers.spearmint.spearmint_gitfork_mod_parser")


def manipulate_config(config):
    # special cases
    if not config.has_option('SPEARMINT', 'method'):
        raise Exception("SPEARMINT:method not specified in .cfg")
    if not config.has_option('SPEARMINT', 'method_args'):
        raise Exception("SPEARMINT:method-args not specified in .cfg")

    # GENERAL
    if not config.has_option('SPEARMINT', 'max_finished_jobs'):
        config.set('SPEARMINT', 'max_finished_jobs',
                   config.get('HPOLIB', 'number_of_jobs'))

    path_to_optimizer = config.get('SPEARMINT', 'path_to_optimizer')
    print path_to_optimizer
    if not os.path.isabs(path_to_optimizer):
        path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

    path_to_optimizer = os.path.normpath(path_to_optimizer)
    if not os.path.exists(path_to_optimizer):
        logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
        sys.exit(1)

    config.set('SPEARMINT', 'path_to_optimizer', path_to_optimizer)

    if not config.has_option('METALEARNING', 'tasks'):
        raise Exception("METALEARNING:tasks not specified in config.cfg")
    if not os.path.exists(config.get('METALEARNING', 'tasks')):
        raise Exception('Tasks file %s does not exist' % config.get('METALEARNING', 'tasks'))

    if not config.has_option('METALEARNING', 'experiments'):
        raise Exception("METALEARNING:experiments not specified in config.cfg")
    if not os.path.exists(config.get('METALEARNING', 'experiments')):
        raise Exception('Experiments file %s does not exist' % config.get('METALEARNING', 'experiments'))

    return config
