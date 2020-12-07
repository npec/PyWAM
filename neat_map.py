# ----------------------------------------------------------------------------------------------------------------------
# Name:        main
# Purpose:     NEAT MAP
#
# Author:      Eligio
#
# Created:     02/11/2020
# Copyright:   (c) Eligio 2019
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

import numpy as np
from netCDF4 import Dataset

from logger_config import logger_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = logger_config(logger=logger)


def neatvar(nc: Dataset):
    """

    :param nc:
    :return:
    """
    dim = nc.createVariable('neat', 'int8',
                            (u'lat', u'lon'),
                            zlib=True, complevel=6)
    dim.setncatts({
        'standard_name': 'NOWPAP Eutrophication Assessment Map',
        'long_name': 'classes of eutrophication potential with 6 '
                     'being the worst condition and 1 the best',
        'units': 'classes',
        '_FillValue': np.int8(-128),
        'valid_min': np.int8(1),
        'valid_max': np.int8(6),
    })


def nccopy(src: str, trg: str, data: np.ma.array):
    with Dataset(src, 'r') as ncs, \
            Dataset(trg, 'r+') as nct:
        # Glob attrs
        nct.setncattrs({ncs.getncattr(at) for at in ncs.ncattrs()})
        # Dims
        for name in ncs.dimensions:
            nct.createDimension(
                ncs.dimensions[name]
            )
            nct[name][:] = ncs[name][:]
        # The variable
        neatvar(nc=nct)
        nct['neat'][:] = data.astype(np.int8)


def map_gen(trend_file: str, comp_file: str, ofile: str, threshold: float = 5):
    with Dataset(comp_file, 'r') as ncf, \
            Dataset(trend_file, 'r') as trn:
        slope = trn['Sen_slope_90'][:]
        try:
            comp = ncf['Chl'][:]
        except IndexError:
            comp = ncf['chlor_a'][:]

        msv = 128 * trn['Sen_slope_90'].getncattr(
            'scale_factor') + trn['Sen_slope_90'].getncattr('add_offset')

        slope[np.ma.where(slope == msv)] = 0
        data = np.ma.zeros(shape=comp.shape,
                           fill_value=-128,
                           dtype=np.int8)

        # LD: 1 MEAN_CHL < 5, Slope < 0
        data[(comp < threshold) &
             (slope < 0)] = 1

        # LN: 2 MEAN_CHL < 5, Slope = 0
        data[(comp < threshold) &
             (slope == 0)] = 2

        # LI: 3 MEAN_CHL < 5, Slope > 0
        data[(comp < threshold) &
             (slope > 0)] = 3

        # HD: 4 MEAN_CHL >= 5, Slope < 0
        data[(comp >= threshold) &
             (slope < 0)] = 4

        # HN: 5 MEAN_CHL >= 5, Slope = 0
        data[(comp >= threshold) &
             (slope == 0)] = 5

        # HI: 6 MEAN_CHL >= 5, Slope > 0
        data[(comp >= threshold) &
             (slope > 0)] = 6
        data.mask = slope.mask

    # Now create our var
    if os.path.isfile(ofile):
        with Dataset(trend_file, 'r+') as nc:
            neatvar(nc=nc)
            nc.variables['neat'][:] = data.astype(np.int8)
    else:
        nccopy(src=trend_file, trg=ofile, data=data)
    return data


def cli_main():
    """Command line parser interface for automated WAM_ANNUAL_MAX and WAM_TREND CLI.
    """
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description='NEAT MAP\n', add_help=True,
                                     epilog=textwrap.dedent('''\
                                     Get the NEAT map from the computation of 
                                     trends (Sen's slope trend) and temporal CHL mean
                                     Type neat_map --help for details.
                                     --------
                                     Examples:
                                        neat_map trend.nc composite.nc -t 3 -o composite.nc
                                        '''
                                                            ),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     )
    parser.add_argument('tfile', help='input file with trend data', type=str, metavar='Trend')
    parser.add_argument("cfile", help="input file with temporal composite data",
                        type=str, metavar="WamComposite")
    parser.add_argument("-t", help="threshold for the NEAT <H | L> CHL (default: 5 mg m^-3)",
                        type=float, dest="threshold", default=5)
    parser.add_argument('-o', help=f"output filename where to save the neat map (default: trend-file')",
                        dest='ofile', type=str)

    opts = parser.parse_args()
    if opts.save is None:
        opts.save = opts.tfile

    map_gen(
        trend_file=opts.tfile,
        comp_file=opts.cfile,
        threshold=opts.threshold,
        ofile=opts.ofile
    )

    time_elapsed = (time.perf_counter() - start)
    hrs = int(time_elapsed // 3600)
    mnt = int(time_elapsed % 3600 // 60)
    sec = int(time_elapsed % 3600 % 60)
    logger.info(f'Processing Time:{hrs:3} hrs'
                f'{mnt:3} min.{sec:3} sec.')


if __name__ == '__main__':
    cli_main()
