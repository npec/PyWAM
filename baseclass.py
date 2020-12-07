# ----------------------------------------------------------------------------------------------------------------------
# Name:        baseclass
# Purpose:     Methods for the NEAT
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
import os
import subprocess
from abc import (ABC, abstractmethod)
from glob import glob

from dateutil.parser import parse
from netCDF4 import Dataset
from pyhdf.SD import (SD, SDC)
import numpy as np


class BaseClass(ABC):
    def __init__(self, ipat: str, **kwargs):
        # twd: str, syear: int,
        #          eyear: int, logger, verbose: bool, debug: bool):
        self.eyear = kwargs.pop('eyear')
        self.syear = kwargs.pop('syear')
        self.logger = kwargs.pop('logger')
        self.verbose = kwargs.pop('verbose')
        self.ipat = ipat
        self.debug = kwargs.pop('debug')
        self.twd = kwargs.pop('twd')

    def get_year(self):
        yy = []
        if self.verbose:
            self.logger.info('Reading time from files')

        if self.ipat.endswith('.nc'):
            for i, f in enumerate(glob(self.ipat)):
                if self.debug:
                    self.logger.info(f'{i:<4}: {f}')
                else:
                    print('.', end='')
                with Dataset(f, 'r') as nc:
                    yy.append(
                        parse(nc.getncattr(
                            'time_coverage_start')).year
                    )
        if self.ipat.endswith('.hdf'):
            for i, f in enumerate(glob(self.ipat)):
                if self.debug:
                    self.logger.info(f'{i:<4}: {f}')
                else:
                    print('.', end='')
                hdf = SD(f, SDC.READ)
                yy.append(
                    parse(hdf.attributes()[
                              'time_coverage_start']).year
                )
                hdf.end()
        print(' ')
        return np.array(yy)

    def get_files(self):
        files = []
        if 'year' in self.ipat:
            for y in range(self.syear,
                           self.eyear + 1):
                files.extend(glob(
                    self.ipat.format(year=y)))
        else:
            self.ipat = self.strip_digit(
                file=glob(self.ipat)[0])

            if self.debug:
                self.logger.info(f'Pattern: {self.ipat}')
            for y in range(self.syear,
                           self.eyear + 1):
                files.extend(glob(
                    self.ipat.format(year=y)))
        return files

    @abstractmethod
    def nc2hdf(self, files: list):
        raise NotImplementedError

    @abstractmethod
    def hdf2nc(self):
        raise NotImplementedError

    def copy(self, files: list, dst: str = None):
        """
        Move yearRange files to tempDir
        :param dst:
        :param list files: input files to move
        :return: tempDir
        """
        if dst is None:
            dst = self.twd

        for src in files:
            cmd = f'copy /y {src} {dst} > nul'
            self.shell(cmd=cmd)
        return dst

    def mkdir(self):
        """
        Create temporary dir to hold yearRange files
        :return: tempDir
        """
        if not os.path.isdir(self.twd):
            os.makedirs(self.twd)
        return self.twd

    def strip_digit(self, file: str):
        bsf = os.path.basename(file)
        bsd = os.path.dirname(file)

        pos = []
        for i, char in enumerate(bsf):
            if char.isdigit():
                pos.append(i)
        if self.debug:
            self.logger.info(f'Digits: {pos}')

        if (len(pos) == 0) or ('trend_sen' in bsf):
            pos = []
            for i, s in enumerate(bsf):
                if s == '_': pos.append(i)
            idx = np.where(np.diff([pos[0]] + pos) == 1)[0]
            if idx.size > 0:
                pos = [pos[0], pos[idx[-1]]]
            else:
                pos = [pos[0]]

            if self.debug:
                self.logger.info(f'Underscore: {pos}')

        if min(pos) == 0:
            return f'{bsd}{os.sep}' \
                   f'{{year}}*{bsf[max(pos) + 1:]}'

        return f'{bsd}{os.sep}' \
               f'{bsf[:min(pos)]}{{year}}*' \
               f'{bsf[max(pos) + 1:]}'

    def shell(self, cmd: str, cwd: bool = False):
        if cwd is True:
            status = subprocess.call(
                cmd, shell=True, cwd=self.twd
            )
        else:
            status = subprocess.call(
                cmd, shell=True
            )

        if self.debug:
            self.logger.info(cmd)
        if status != 0:
            self.logger.exception(
                f"Command failed with return code: {status}"
            )
        return status

    def subproc(self, cmd: list):
        sub = subprocess.Popen(
            cmd, cwd=self.twd
        )
        sub.wait()
        return sub
