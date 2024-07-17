import os
path = os.path.dirname(__file__)

# Make sure the env var exists
try:
    os.environ['LD_LIBRARY_PATH']
except (KeyError):
    os.environ['LD_LIBRARY_PATH'] = ''

os.environ['LD_LIBRARY_PATH'] = f"{str(os.environ['LD_LIBRARY_PATH'])}:{path}"

class CupybaraPaths:
    cupybara_libs = f'{path}/lib/libcupybara.so'
