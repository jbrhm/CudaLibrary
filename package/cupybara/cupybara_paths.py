import os
path = os.path.dirname(__file__)
if not os.environ['PATH'].__contains__(path):
  os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
if not os.environ['LD_LIBRARY_PATH'].__contains__(path):
  os.environ['LD_LIBRARY_PATH'] = path + os.pathsep + os.environ['LD_LIBRARY_PATH']

class CupybaraPaths:
    cupybara_libs = f'{path}/lib/libcupybara.so'
