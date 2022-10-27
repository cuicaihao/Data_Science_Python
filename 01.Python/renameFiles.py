#! python3
# renameDates.py - Renames filenames with American MM-DD-YYYY date format
# to European DD-MM-YYYY.

import argparse
import shutil, os, re

# Create a regex that matches files with the American date format.
datePattern = re.compile(r"""^([0-9]{2})_(.*)_([0-9]{2,4})(\.pdf)$""")

parser = argparse.ArgumentParser(description='Rename some files.')
parser.add_argument('--dir',
                    metavar='N',
                    type=str,
                    nargs='+',
                    help='an folder of files need to be renamed')
parser.add_argument('--output',
                    metavar='N',
                    type=str,
                    nargs='+',
                    help='an folder save renamed files')
args = parser.parse_args()
# print(args.dir)

# TODO: Loop over the files in the working directory.
for amerFilename in os.listdir(args.dir[0]):
    mo = datePattern.search(amerFilename)
    # print(amerFilename, mo)
    # TODO: Skip files without a date.
    if mo == None:
        continue
    else:
        # TODO: Get the different parts of the filename.
        beforePart = mo.group(1)
        middlePart = mo.group(2)
        afterPart = mo.group(3)

    print(beforePart, middlePart, afterPart)

    if '22' in afterPart:
        afterPart = '2022'

    # TODO: Form the new-style filename.
    newFilename = afterPart + '_' + middlePart + '_' + beforePart + '.pdf'

    # TODO: Get the full, absolute file paths.
    absWorkingDir = os.path.abspath(args.dir[0])
    amerFilename = os.path.join(absWorkingDir, amerFilename)

    absWorkingOutput = os.path.abspath(args.output[0])
    newFilename = os.path.join(absWorkingOutput, newFilename)

    # TODO: Rename the files.
    print('Renaming "%s" to "%s"...' % (amerFilename, newFilename))
    # shutil.move(amerFilename, newFilename)  # uncomment after testing
    shutil.copyfile(amerFilename, newFilename)  # uncomment after testing