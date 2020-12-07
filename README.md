## NOWPAP Eutrophication Assessment Tool (NEAT) Map Generator 

This Python-based software is dedicated to generating the NEAT map through the use of WIMSOFT.
This tool is just an integrator of the many command line (CLI) WAM tools 

### Setup of Python Environment
---
The only Python setup required is just the installation of the Python itself and a few libraries 
used for number crunching and image printing.

#### Prerequisites
##### Required Modules
The software have been tested with the following module versions

- netCDF4==1.5.5
- numpy==1.20.0rc1
- matplotlib==3.1.3
- python-dateutil==2.8.1
- pyhdf==0.10.2
- coloredlogs==14.0
Aside Python modules, [WIMSOFT](https://wimsoft.com/) needs to be installed in order to have the scripts working

##### Running the main
- Once all is set, inside the ```main.py``` change the following parameters according to your environment.
  - fp - input file pattern that includes the full path
  - op - output path 
  - pp - path where to output the png images generated
  - if - input files, those consistent of the trend file from wam_trend, the composite file, the count, etc.
  - dv - data variable. This is the name of the variable inside "if" above
  - by - the base year, i.e., the year in which the trend estimates start

- The above parameter stores sensor-based input parameters
For any function call within this tool, the input parameters or the user case can be infered through command line 
  interface. All scripts are CLI and have the right documentation within
## Authors

Eligio Maure <maure@npec.or.jp>

## License
[MIT](https://opensource.org/licenses/MIT)
