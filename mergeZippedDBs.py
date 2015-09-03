#!/usr/bin/env python

"""
    This utility calls the combineEbeDatabasesFromZippedResults to merge 
    the collected database and particle information database within the zipped 
    files.
"""
from sys import argv, exit, stdout
from os import path, listdir
from subprocess import call

# get options
try:
    parentFolder = path.abspath(argv[1])
    numberOfZipFiles = int(argv[2])
    # get optional parameters
    if len(argv)>=4:
        subfolderPattern = argv[3]
    else:
        subfolderPattern = "job-(\d*).zip"
    if len(argv)>=5:
        databaseFilename = argv[4]
    else:
        databaseFilename = "collected.db"
    if len(argv)>=6:
        databaseFilename_particles = argv[5]
    else:
        databaseFilename_particles = "particles.db"
except:
    print("Usage: mergeZippedDBs.py parent_folder expected_number_of_zip_files [subfolder_pattern] [database_filename]")
    exit()


call("python ./combineEbeDatabasesFromZippedResults.py %s %s" % (parentFolder, databaseFilename), shell=True)
call("python ./combineEbeDatabasesFromZippedResults.py %s %s" % (parentFolder, databaseFilename_particles), shell=True)