# ----------------------------------------------------------------------------------------------------------------------
# Name:        ANNUAL MAX
# Purpose:     CLI for the generation of annual max based on WIM WAM CLT
#
# Author:      Eligio
#
# Created:     30/10/2020
# Copyright:   (c) Eligio 2020
# Licence:     <MIT>
#
# Comments/questions:
#   E. R. Maure, maure@npec.or.jp
# ----------------------------------------------------------------------------------------------------------------------
import argparse
import logging
import os
import re
import subprocess
import textwrap
import time
from glob import glob

import numpy as np
from netCDF4 import Dataset

from baseclass import BaseClass
from logger_config import logger_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = logger_config(logger=logger)
__version__ = '1.0.0'


class WamAnnulMaxTrend(BaseClass):
    __slots__ = 'debug', 'eyear', 'ipat', 'ipath', 'median', 'opath', \
                'reduce', 'sig', 'syear', 'twd', 'type', 'verbose'

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
        self.sig = kwargs.pop('sig')
        if self.sig is None:
            self.sig = 90
        self.median = kwargs.pop('median')
        self.reduce = kwargs.pop('reduce')
        self.type = kwargs.pop('type')
        if self.type is None:
            self.type = 'Sen'
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

    def annual_max(self):
        """

        :return:
        """
        if self.verbose:
            logger.info(f'Annual Max: {self.syear}:'
                        f'{self.eyear}\n{"=" * 20}')
        ipat = self.ipat.replace('{year}', '')
        if 'L3BMax' in self.twd:
            return 'A*.L3m_MAX_CHL.hdf'
        else:
            status = subprocess.check_output(['wam_annual_max',
                                              os.path.basename(ipat)],
                                             cwd=self.twd, shell=True)
            result = status.decode('utf-8')
            logger.info(f"status: {result}")
            file = re.findall('.*_Max.hdf', ''.join(result))
            return file[0].split('Annual max in ')[-1]

    def fix_slope(self, files: list):
        if self.debug:
            logger.info(f'Files: {files}')

        slope = re.findall('.*_trend_sen_90.nc', '\n'.join(files))

        count = re.findall('.*_ValidCounts.nc', '\n'.join(files))
        if len(count) == 0:
            count = glob(f'{os.path.dirname(os.path.dirname(slope[0]))}'
                         f'\\A*.L3m_MAX_CHL.nc')[0]
            cmd = f'copy /y {count} {self.twd} > nul'
            self.shell(cmd=cmd)
            count = [f'{self.twd}\\{os.path.basename(count)}']

        try:
            with Dataset(slope[0], 'r+') as slp, \
                    Dataset(count[0], 'r+') as cnt:
                slp['Sen_slope_90'].renameAttribute('Slope', 'scale_factor')
                slp['Sen_slope_90'].renameAttribute('Intercept', 'add_offset')
                slp['Sen_slope_90'].delncattr('valid_min')
                slp['Sen_slope_90'].delncattr('valid_max')

                slp['lon'][:] = cnt['lon'][:]
                slp['lat'][:] = cnt['lat'][:]

                sds0 = slp['Sen_slope_90'][:]
                var = [key for key in cnt.variables.keys()
                       if (('lon' not in key.lower()) or
                           ('lat' not in key.lower()))][-1]
                cnt[var].delncattr('valid_min')
                cnt[var].delncattr('valid_max')

                sds1 = cnt[var][:]
                sds1 = np.ma.masked_where(sds1 == 0, sds1)

                sds0 = np.ma.masked_where(sds1.mask, sds0)
                slp['Sen_slope_90'][:] = sds0

        except RuntimeError as er:
            logger.exception('FixSlopeError', exc_info=er)
            pass

    def get_year(self):
        return super().get_year()

    def nc2hdf(self, files: list):
        for f in files:
            self.subproc(cmd=['wam_nc2hdf', f])
        fp = os.path.basename(
            self.ipat.replace('.nc', '.hdf')).split('{year}*')
        fp = '*'.join(fp)
        if self.debug:
            logger.info(fp)
        return glob(f'{self.twd}{os.sep}{fp}')

    def hdf2nc(self, file: str = None):
        files = glob(
            f'{self.twd}{os.sep}*.hdf'
        )
        rf = []
        for f in files:
            self.subproc(cmd=['wam_hdf2nc', f])
            cmd = f'del /f {f}'
            self.shell(cmd=cmd)
            rf.append(self.rename(file=f))
        return rf

    def mkdir(self):
        return super().mkdir()

    def copy(self, files: list, dst: str = None):
        return super().copy(files=files, dst=dst)

    def rename(self, file: str):
        new = self.strip_digit(
            file=file.replace('.hdf', '.nc')
        )
        dst = new.format(
            year=f'{self.syear}{self.eyear}_'
        ).replace('*', '')
        cmd = f"rename {file.replace('.hdf', '.nc')} " \
              f"{os.path.basename(dst)}"
        self.shell(cmd=cmd)
        return dst

    def strip_digit(self, file: str):
        return super().strip_digit(file=file)

    def shell(self, cmd: str, cwd: bool = False):
        return super().shell(cmd=cmd, cwd=cwd)

    def subproc(self, cmd: list):
        return super().subproc(cmd=cmd)

    def trend(self, file: str):
        if self.median is not None:
            if self.reduce is not None:
                if self.verbose:
                    logger.info(f'Trend | {self.type}: {self.syear}:'
                                f'{self.eyear}\n{"=" * 19}')
                cmd = ['wam_trend', file,
                       f'type={self.type}',
                       f'median={self.median}',
                       f'reduce={self.reduce}',
                       f'sig={self.sig}']
            else:
                if self.verbose:
                    logger.info(f'Trend | {self.type}: {self.syear}:'
                                f'{self.eyear}\n{"=" * 19}')
                cmd = ['wam_trend', file,
                       f'median={self.median}',
                       f'type={self.type}',
                       f'sig={self.sig}']
        else:
            if self.reduce is not None:
                if self.verbose:
                    logger.info(f'Trend | {self.type}: {self.syear}:'
                                f'{self.eyear}\n{"=" * 19}')
                cmd = ['wam_trend', file,
                       f'type={self.type}',
                       f'reduce={self.reduce}',
                       f'sig={self.sig}']
            else:
                if self.verbose:
                    logger.info(f'Trend | {self.type}: {self.syear}:'
                                f'{self.eyear}\n{"=" * 19}')
                cmd = ['wam_trend', file,
                       f'type={self.type}',
                       f'sig={self.sig}']
        sub = self.subproc(cmd=cmd)
        return sub

    def get(self):
        """

        :return:
        """
        files = super().get_files()
        if self.debug:
            logger.info('\n'.join(files))

        # make tempDir to hold compTxt file
        self.mkdir()

        # Move files corresponding to yearRange
        if self.ipat.endswith('.nc'):
            files = self.nc2hdf(files=files)
            self.ipat = self.ipat.replace('.nc', '.hdf')
        else:
            self.copy(files=files)

        # Get annual max
        file = self.annual_max()

        # Compute trend
        if self.verbose:
            logger.info(f'TrendFile: {file}')
        self.trend(file=file)

        # Remove copied files to tempDir
        [self.shell(
            cmd=f'del /f {self.twd}{os.sep}'
                f'{os.path.basename(f)} > nul'
        ) for f in files]

        # convert to netCDF and rename
        files = self.hdf2nc()
        if self.debug:
            logger.info(files)

        # fix with slope
        self.fix_slope(files=files)

        # move files to final dest
        [self.shell(
            cmd=f'move {f} {self.opath} > nul'
        ) for f in files]

        # eliminate temp dir
        cmd = f'rmdir /q/s {self.twd}'
        self.shell(cmd=cmd)

        file = re.findall(
            '.*_Max_trend_sen_90.nc', '\n'.join(
                [os.path.basename(f) for f in files])
        )
        if len(file) == 0:
            file = re.findall(
                '.*_trend_sen_90.nc', '\n'.join(
                    [os.path.basename(f) for f in files])
            )
        return f'{self.opath}{os.sep}{file[0]}'


def cli_main():
    """Command line parser interface for automated WAM_ANNUAL_MAX and WAM_TREND CLI.
    """
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=f'WAM_ANNUAL_MAX&WAM_TREND ver. {__version__}'
                                                 '\n', add_help=True,
                                     epilog=textwrap.dedent('''\
                                     Get Annual Max over a file collection
                                     and calculates the Sen's slope trend
                                     Type annual_max --help for details.
                                     --------
                                     Examples:
                                        annual_max A*CHL.nc -s 2003 -e 2018 
                                        annual_max A*CHL.hdf -o DIR
                                        '''
                                                            ),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     )
    parser.add_argument('ipat', help='input pattern', type=str, metavar='Pattern')
    parser.add_argument('-v', help=f"verbose <0: False | 1: True> (default: {1}')",
                        dest='verbose', type=int, default=1)
    parser.add_argument('-o', help=f"output path (default: {os.path.dirname(os.getcwd())}')",
                        dest='opath', type=str, default=os.path.dirname(os.getcwd()))
    parser.add_argument("-s", help="start year <yyyy> (default: None)", type=int, dest="syear")
    parser.add_argument("-e", help="end year <yyyy> (default: None)", type=int, dest="eyear")
    parser.add_argument("-m", help="median <3, 5,..., N> applies median \n"
                                   "filter of size N to all images (default: None)",
                        type=int, dest="median")
    parser.add_argument("-r", help="reduce <2, 3, 4,..., M>\n"
                                   "reduces the image size by M times (default: None)",
                        type=int, dest="reduce")
    parser.add_argument("-t", help="trend type <Sen | Lin> Lin is regression, "
                                   "and Sen is nonparametric Sen's slope (default: Sen)",
                        default='Sen', type=str, dest="type")
    parser.add_argument("-ci", help="confidence interval for the slope ci <90 | 95 | 99>\n"
                                    "correspond to 0.1, 0.05 and 0.01 of the two-sided normal\n"
                                    "distribution. For Linear, CI can be anything from 50 and 99"
                                    "(default: 90)",
                        default=90, type=float, dest='ci'
                        )
    opts = parser.parse_args()
    wamt = WamAnnulMaxTrend(
        ipat=opts.ipat, **{'opath': opts.opath,
                           'syear': opts.syear,
                           'eyear': opts.eyear,
                           'median': opts.median,
                           'reduce': opts.reduce,
                           'type': opts.type,
                           'sig': opts.ci,
                           'verbose': bool(opts.verbose),
                           'debug': False
                           })
    wamt.get()

    time_elapsed = (time.perf_counter() - start)
    hrs = int(time_elapsed // 3600)
    mnt = int(time_elapsed % 3600 // 60)
    sec = int(time_elapsed % 3600 % 60)
    logger.info(f'Processing Time:{hrs:3} hrs'
                f'{mnt:3} min.{sec:3} sec.')


if __name__ == '__main__':
    cli_main()
