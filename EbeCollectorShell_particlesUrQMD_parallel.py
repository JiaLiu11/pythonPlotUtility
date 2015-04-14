#!/usr/bin/env python
"""
    This is the collect events shell to collect hybrid run after parallel
    implementation of osc2u+UrQMD. It first generate a database for each 
    parallel node result, then merge all databases together, with the understanding
    that they all come from the same hydro event.
"""

from sys import argv, exit
from os import path, rename

try:
    from_folder = path.abspath(argv[1])
except:
    print("Usage: shell from_folder num_of_split [urQMD_output_pattern] [sub_folder_pattern] [database_filename]")
    exit()

try:
    num_of_split = int(argv[2])
except:
    print("Usage: shell from_folder num_of_split [urQMD_output_pattern] [sub_folder_pattern] [database_filename]")
    exit()

# get optional parameters
if len(argv)>=4:
    urQMD_output_pattern = argv[3]
else:
    urQMD_output_pattern = "particle_list_%d.dat"
if len(argv)>=5:
    subfolder_pattern = argv[4]
else:
    subfolder_pattern = "event-(\d*)"
if len(argv)>=6:
    database_filename = argv[5]
else:
    database_filename = "particles.db"

# call EbeCollector
from EbeCollector import EbeCollector

# loop over all urqmd results
print "Start to collect data from all subfolders in %s/%s"%(from_folder,subfolder_pattern)
for inode in range(num_of_split):
    db_name_now = "node_%d.db"%inode
    # rename current source file to let EbeCollect find it
    db_source_file = path.join(from_folder, "event-1", urQMD_output_pattern%inode)
    db_source_file_renamed = path.join(from_folder, "event-1", "particle_list.dat")
    rename(db_source_file, db_source_file_renamed)
    EbeCollector().collectParticleinfo(from_folder, subfolder_pattern, 
                                       databaseFilename = db_name_now,
                                       rap_range = (-1.5, 1.5),
                                       particles_to_collect = ['allUrqmd'])
    # reverse the name of source file
    rename(db_source_file_renamed, db_source_file)
    print "Data from node %s has been collected!"%db_source_file.split('/')[-1]

# merge all subsequent databases to database 1
from DBR import SqliteDB
collector=EbeCollector()
toDB = SqliteDB(path.join("./", "node_0.db"))
for inode in range(1, num_of_split):
    fromDB = SqliteDB(path.join("./", "node_%d.db"%inode))
    # merge two database
    collector.mergeparticle_parallel_Databases(toDB, fromDB)
    # delete merged databse
    fromDB.deleteDatabase(True)

# rename database
rename(path.join("./", "node_0.db"),
       path.join("./", database_filename))

print "All database merged to %s"%database_filename


