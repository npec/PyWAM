# ----------------------------------------------------------------------------------------------------------------------
# Name:        Composite
# Purpose:     CLI for the generation of temporal composite based on WIM WAM CLT
#
# Author:      Eligio
#
# Created:     31/10/2020
# Copyright:   (c) Eligio 2020
# Licence:     <MIT>
#
# Comments/questions:
#   E. R. Maure, maure@npec.or.jp
# ----------------------------------------------------------------------------------------------------------------------
import argparse
import logging
import os
import textwrap
import time
from glob import glob

import numpy as np

from baseclass import BaseClass
from logger_config import logger_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = logger_config(logger=logger)

__version__ = '1.0.0'


class WamComposite(BaseClass):

    def __init__(self, ipat: str, **kwargs):
        kwargs['twd'] = f'{os.path.dirname(ipat)}' \
                        f'{os.sep}wam_temp'
        kwargs['logger'] = logger
        super().__init__(ipat, **kwargs)

        self.ipath = os.path.dirname(ipat)
        self.eyear = kwargs.pop('eyear')
        self.syear = kwargs.pop('syear')
        self.opath = kwargs.pop('opath')

        if self.opath is None:
            self.opath = os.path.dirname(os.getcwd())
        self.verbose = kwargs.pop('verbose')
        self.debug = kwargs.pop('debug')

        if (self.syear is None) or \
                (self.eyear is None):
            try:
                yy = self.get_year()
                if self.syear is None:
                    self.syear = np.min(yy)
                if self.eyear is None:
                    self.eyear = np.max(yy)
            except Exception as exc:
                logger.exception(
                    f'{exc}\nCannot get time from files\n'
                    f'Please specify -s <yyyy> -e <yyyy>')
                raise exc

    def get_year(self):
        return super().get_year()

    def nc2hdf(self, files: list):
        for f in files:
            self.subproc(cmd=['wam_nc2hdf', f])

        ifile = self.temp_file(
            files=glob(f'{self.twd}{os.sep}*.hdf')
        )
        return ifile

    def hdf2nc(self, file: str = None):
        self.subproc(
            cmd=['wam_hdf2nc', file],
        )
        cmd = f'del /f {file}'
        self.shell(cmd=cmd)
        return file.replace('.hdf', '.nc')

    def copy(self, files: list, dst: str = None):
        return super().copy(files=files, dst=dst)

    def strip_digit(self, file: str):
        return super().strip_digit(file=file)

    def shell(self, cmd: str, cwd: bool = False):
        return super().shell(cmd=cmd, cwd=cwd)

    def temp_file(self, files: list):
        """

        :param files:
        :return:
        """
        ifile = f'{self.twd}{os.sep}' \
                f'{self.syear}{self.eyear}.txt'
        lines = '\n'.join(files)
        with open(ifile, 'w') as txt:
            txt.writelines(f'{lines}\n')
        return ifile

    def get(self):
        # Get input files
        files = super().get_files()

        # make tempDir to hold compTxt file
        self.mkdir()

        if self.ipat.endswith('.nc'):
            ifile = self.nc2hdf(files=files)
            # self.ipat = self.ipat.replace('.nc', '.hdf')
        else:
            # Save files to input text file
            ifile = self.temp_file(files=files)

        # Get the initial name string
        init = self.strip_digit(file=files[0])
        init = os.path.basename(init).split('{year}')[0]

        # WIM composite command/run
        ofile = f'{self.opath}{os.sep}{init}' \
                f'{self.syear}{self.eyear}_comp.hdf'
        cmd = f'wam_composite {ifile} {ofile}'
        self.shell(cmd=cmd)

        # convert to netcdf
        file = self.hdf2nc(file=ofile)

        # Move to file dest
        cmd = f'move {self.twd}{os.sep}' \
              f'{os.path.basename(file)} ' \
              f'{self.opath} > nul'
        self.shell(cmd=cmd)

        # eliminate temp dir
        cmd = f'rmdir /q/s {self.twd}'
        self.shell(cmd=cmd)

        return f'{self.opath}{os.sep}' \
               f'{os.path.basename(file)}'


def formatter():
    # ArgumentDefaultsHelpFormatter
    return argparse.RawDescriptionHelpFormatter


def cli_main():
    """Command line parser interface for automated WAM_COMPOSITE.
       """
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=f'WAM_COMPOSITE ver. {__version__}'
                                                 '\n', add_help=True,
                                     epilog=textwrap.dedent('''\
                                        Get WamComposite over a given period.
                                        Alternatively WAM_COMPOSITE_INTIME can be used
                                        Type composite --help for details.
                                        --------
                                        Examples:
                                           composite A*CHL.nc 2015 2018 
                                           composite A*CHL.hdf 2015 2018 -o DIR
                                           '''
                                                            ),
                                     formatter_class=formatter(),
                                     )
    parser.add_argument('ipat', help='input pattern', type=str, metavar='Pattern')
    parser.add_argument("syear", help="composite start year <yyyy>",
                        type=int, metavar="Start")
    parser.add_argument("eyear", help="composite end year <yyyy>",
                        type=int, metavar="End")
    parser.add_argument('-v', help=f"verbose <0: False | 1: True> (default: {1}')",
                        dest='verbose', type=int, default=1)
    parser.add_argument('-o', help=f"output path (default: {os.path.dirname(os.getcwd())}')",
                        dest='opath', type=str, default=os.path.dirname(os.getcwd()))
    opts = parser.parse_args()

    wamc = WamComposite(
        ipat=opts.ipat, **{'opath': opts.opath,
                           'syear': opts.syear,
                           'eyear': opts.eyear,
                           'verbose': bool(opts.verbose),
                           'debug': False
                           })
    wamc.get()

    time_elapsed = (time.perf_counter() - start)
    hrs = int(time_elapsed // 3600)
    mnt = int(time_elapsed % 3600 // 60)
    sec = int(time_elapsed % 3600 % 60)
    logger.info(f'Processing Time:{hrs:3} hrs'
                f'{mnt:3} min.{sec:3} sec.')


if __name__ == '__main__':
    cli_main()
