# Copyright (c) 2022, Yangjie Zhou.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import logging.handlers
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(CURRENT_DIR, "logs")
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
###################### configure the log format ###############################################
# Configure main log format
logger = logging.getLogger('GNNTester')
logger.setLevel(logging.DEBUG)

# A handler class which writes formatted logging records to disk files.
fh_info_path = os.path.join(LOG_DIR,"GNNTester.txt")
fh_err_path = os.path.join(LOG_DIR,"GNNTester.txt")
# maxBytes = 5MB
fh_info = logging.handlers.RotatingFileHandler(fh_info_path,maxBytes=5*1024*1024, backupCount=5)
fh_err = logging.handlers.RotatingFileHandler(fh_err_path,maxBytes=5*1024*1024, backupCount=5)

# create another handler, for stdout in terminal
# A handler class which writes logging records to a stream
sh = logging.StreamHandler()
sh.setLevel(logging.ERROR)

# set formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh_info.setFormatter(formatter)
fh_err.setFormatter(formatter)
sh.setFormatter(formatter)

# set filter
info_filter = logging.Filter()
info_filter.filter = lambda record: record.levelno <= logging.WARNING
err_filter = logging.Filter()
err_filter.filter = lambda record: record.levelno > logging.WARNING
fh_info.addFilter(info_filter)
fh_err.addFilter(err_filter)

# add handler to logger
logger.addHandler(fh_info)
logger.addHandler(fh_err)
logger.addHandler(sh)
