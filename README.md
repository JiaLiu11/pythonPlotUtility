pythonPlotUtility
=================

some scripts for plotting results from database from iEBE package with matplotlib 

## Pipeline for collecting data from URQMD format and make sense of it:

### 1. collect raw data to preliminary data base
This step collects URQMD output from .dat files to a SQLite database file, with default 
name "particles.db". The .dat files should be put under a folder with the name event-%d, to 
make it recognizable by the collector. Suppose we have one such folder with the name event-1 in the current folder, the following line collect data from it and write out a database at the current folder.

> ./EbeCollectorShell_particlesUrQMD.py ./

### 2. convert collected data to a nicer form
This step is a bridge step to covert raw data in the database to more interesting data. It 
generates a new and smaller SQLite file with the default name "analyzed_particles.db". The
converter "particleReader.py" will reorganize the tables in the database and sort particles by rapidity etc. Simply run the shell to do it. 

> ./particleReaderShell.py particles.db

### 3. analyze the database
The script "AnalyzedEventsReader.py" contains many functions which can calculate and output flow, spectra etc. Write what you want to know at the end of this file, and excute it.

> python AnalyzedEventsReader.py analyzed_particles.db

