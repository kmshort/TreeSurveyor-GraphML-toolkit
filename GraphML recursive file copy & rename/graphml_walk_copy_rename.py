# !/usr/bin/python
import os
from shutil import copyfile

# this script copys all files with extension SUFFIX from a directory tree, in the process renaming them based on the directory name from RENAME_LEVEL up thedirectory tree

#directory = "."
directory = os.getcwd()
fileFound = False
rename_level = 3

suffix = raw_input("Please enter extension of files you want to copy:  ")

for root, dirs, files in os.walk(directory):
        for file in files:
            if (file.endswith(suffix)):
                
                directory_split = root.split('\\')
                #grab directory name from 3 levels back
                newfilename = directory_split[-(rename_level)] + suffix
                
                print "source", os.path.join(root,file)
                print "destination", os.path.join(directory,newfilename)
                
                copyfile(os.path.join(root,file), os.path.join(directory,newfilename))
                fileFound = True
                
if not fileFound:
    print "No", suffix, "files found"