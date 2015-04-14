#!/usr/bin/env python
"""
    This is one of the shells to the EbeCollector class. This one
    creates a database using data from subfolders containing multiple
    UrQMD events.
"""

from sys import argv, exit
from os import path

try:
    from_folder = path.abspath(argv[1])
except:
    print("Usage: shell from_folder [sub_folder_pattern] [database_filename]")
    exit()

# get optional parameters
if len(argv)>=3:
    subfolder_pattern = argv[2]
else:
    subfolder_pattern = "event-(\d*)"
if len(argv)>=4:
    database_filename = argv[3]
else:
    database_filename = "particles.db"

# call EbeCollector
from EbeCollector import EbeCollector
EbeCollector().collectParticleinfo(from_folder, subfolder_pattern, 
                                   databaseFilename = database_filename,
                                   rap_range = (-1.5, 1.5),
                                   particles_to_collect = ['allUrqmd'])
