#Copyright (c) 2021 Oleg Kovalenko

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#!/usr/bin/python

import os, subprocess
import json
import time
import sys, getopt
from pathlib import Path

def main(argv):
  start_time = time.time()
  rootdir = os.path.dirname(os.path.realpath(__file__))
  imgdir = rootdir
  cmpfile = os.path.join(rootdir, 'comparisons.json')
  cmpopts = '-metric PSNR'
  try:
    opts, args = getopt.getopt(argv, '', ['rootdir=', 'imgdir=', 'cmpfile=', 'cmpopts='])
  except getopt.GetoptError as err:
    print(err)
    print('rootdir=root path of project, containing the images; relative to this script\'s path. default is path of script.')
    print('imgdir=path from rootdir to the image directory; if rootdir is also given, then rootdir must come first. default is path of script.')
    print('cmpfile=path from rootdir to read or create comparisons file from. default is comparisons.json at the root directory.')
    print('cmpopts=options for magick compare. default is -metric PSNR.')
    sys.exit(2)
  for opt, val in opts:
    print(val)
    if opt == '--rootdir':
      rootdir = os.path.join(os.path.realpath(__file__), val)
    if opt == '--imgdir':
      imgdir = os.path.join(rootdir, val)
    if opt == '--cmpfile':
      cmpfile = os.path.join(rootdir, val)
    if opt == '--cmpopts':
      cmpopts = val
  cmd = 'magick compare {0} {1} {2} null:'
  
  data = {}
  if (os.path.exists(cmpfile)):
    try:
      with open(cmpfile) as f:
        data = json.load(f)
    except json.decoder.JSONDecodeError:
      print('comparison file is not in json format. \'{}\' is the minimum valid json file.')
      
  else:
    with open(cmpfile, 'w+') as f:
      f.write('{}')
    with open(cmpfile) as f:
      data = json.load(f)

  num_compared = 0
  file_compared = len(data)
  
  print('rootdir =', rootdir)
  print('imgdir =', imgdir)
  print('cmpfile =', cmpfile)
  print('cmpopts =', cmpopts)

  each = Path(imgdir).iterdir()
  try:
    for x in each:
      if not x.suffix == '.jpg':
        continue
      after = Path(imgdir).iterdir()
      while True:
        v = ''
        try:
          v = next(after)
        except StopIteration:
          break
        if v == x:
          v = next(after)
          break
           
      for y in after:  
        if not y.suffix == '.jpg':
          continue
        pair = x.name + ', ' + y.name
        if (pair in data):
          continue
        proc = subprocess.Popen(cmd.format(cmpopts, os.path.join(imgdir, x.name), os.path.join(imgdir, y.name)), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, output = proc.communicate()
        data[pair] = output.decode('utf-8')
        num_compared += 1
  
  except KeyboardInterrupt:
    pass
  
  total_time = time.time() - start_time
  print('num compared = ', num_compared)
  print('total compared = ', file_compared + num_compared)
  print('time elapsed = ', total_time)
  tempfilename = os.path.join(os.path.dirname(cmpfile), 'tmp.json')
  with open(tempfilename, 'w') as f:
    json.dump(data, f, indent=2)
  
  os.replace(tempfilename, cmpfile)

if __name__ == '__main__':
  main(sys.argv[1:])