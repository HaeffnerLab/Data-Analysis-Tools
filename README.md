# Data-Analysis-Tools
Python modules for importing data from LabRAD, fitting to data, and some other small things

## How to use
- Add this folder to your python path. One way to do this is to edit `~/.bashrc` and add the line `export PYTHONPATH=$PYTHONPATH:/path/to/this/repo`.
- Configure your `labrad_data_tools` to work for your PC.
  - Create a copy of `labrad_data_tools_config.py.example` and rename it to `labrad_data_tools_config.py`.
  - Set the values of `SCAN_LISTS_DIRECTORY` and `DATA_DIRECTORY` to the directories where scan lists and data are stored on your PC.
  - Set the dict `MULTID_SCAN_MAP` to define the names of 2-dimensional scans that you use and their subscans, as instructed by the commented lines. This allows for functionality like getting a list of all of the subscan scantimes given a superscan scantime.
- Configure your `fitting_functions` by adding fitting functions that are useful to you but are specific to your experiment.
  - Create a copy of `fitting_functions_special.py.example` and rename it to `fitting_functions_special.py`.
  - Define your own functions as instructed in the docstring at the beginning of the file.
  