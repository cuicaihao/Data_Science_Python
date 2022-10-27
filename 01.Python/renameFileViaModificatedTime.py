import argparse
import shutil, os, re
import datetime
import pathlib

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

for amerFilename in os.listdir(args.dir[0]):
    mo = datePattern.search(amerFilename)
    # print(amerFilename)
    absWorkingDir = os.path.abspath(args.dir[0])
    amerFilename = os.path.join(absWorkingDir, amerFilename)
    amerFiletime = os.path.getmtime(amerFilename)
    dt_m = datetime.datetime.fromtimestamp(amerFiletime)
    # print(dt_m, amerFilename)

    f_name = pathlib.Path(amerFilename)
    # get modification time
    m_timestamp = f_name.stat().st_mtime
    # get creation time on windows
    c_timestamp = f_name.stat().st_ctime

    m_time = datetime.datetime.fromtimestamp(m_timestamp)
    c_time = datetime.datetime.fromtimestamp(c_timestamp)
    print(c_time, m_time, amerFilename)