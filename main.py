# ----------------------------------------------------------------------------------------------------------------------
# Name:        main
# Purpose:     main function call for the NEAT
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
import logging
import os
import subprocess
import time
from glob import glob

from netCDF4 import Dataset

from annual_max import WamAnnulMaxTrend
from composite import WamComposite
from figure import GetFigure
from logger_config import logger_config
from neat_map import map_gen

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = logger_config(logger=logger)


def get_trend(ipat: str, outpath: str, syear: int, eyear: int):
    """

    :return:
    """
    wam_trend = WamAnnulMaxTrend(
        ipat=ipat, **{'opath': outpath,
                      'syear': syear,
                      'eyear': eyear,
                      'median': None,
                      'reduce': None,
                      'type': 'Sen',
                      'sig': '90',
                      'verbose': True,
                      'debug': False
                      })
    trend_file = wam_trend.get()
    return trend_file


def get_composite(ipat: str, outpath: str, eyear: int):
    """

    :return:
    """
    cmp = WamComposite(
        ipat=ipat, **{'opath': outpath,
                      'syear': eyear - 2,
                      'eyear': eyear,
                      'verbose': True,
                      'debug': False
                      })
    cmp_file = cmp.get()
    return cmp_file


def get_neat_map(trend_file: str, comp_file: str, threshold: float = 5, ofile: str = None):
    """

    :param trend_file:
    :param comp_file:
    :param threshold:
    :param ofile:
    :return:
    """
    if ofile is None:
        ofile = trend_file
        with Dataset(ofile, 'r') as nc:
            neat_map = 'neat' in nc.variables.keys()
        if neat_map is True:
            return ofile

    map_gen(trend_file=trend_file,
            comp_file=comp_file,
            ofile=ofile,
            threshold=threshold)
    return ofile


def get_figures(sub: str, vid: str, png_file: str, file: str, syear: int, eyear: int):
    logger.info(f'{file}\n{trg_file}\n{"=" * 80}')
    nf = GetFigure(
        file=neat_file,
        var=vid, **{'threshold': None,
                    'caxis': None,
                    'subarea': sub,
                    'cmap': slope,
                    'yrange': [syear, eyear],
                    'verbose': True,
                    'transparent': False,
                    'debug': False,
                    'trg_file': png_file
                    })
    nf.save()


def l3b_comp(ipat: str, outpath: str, eyear: int):
    bsn = os.path.basename(ipat)
    dst = f'{bsn[0]}{eyear - 2}{eyear}.L3m_COMP_CHL.nc'
    if len(glob(f'{outpath}\\{dst}')) > 0:
        return f'{outpath}\\{dst}'

    in_path = os.path.dirname(ipat)
    src = glob(f'{in_path}\\{dst}')[0]
    # dst = f'{outpath}\\{file}'
    cmd = f'copy /y {src} {dst} > nul'
    subprocess.call(cmd, shell=True, cwd=outpath)
    # status
    return f'{outpath}\\{dst}'


def sen_params(sensor: str):
    """
    This file stores sensor-based input parameters
    fp - input file pattern that includes the full path
    op - output path
    pp - path where to output the png images generated
    if - input files, those consistent of the trend file from wam_trend, the composite file, the count, etc.
    dv - data variable. This is the name of the variable inside "if" above
    by - the base year, i.e., the year in which the trend estimates start
    The parameters below are used with the if __name__ == '__main__': of this script. See below.
    For all other function calls, such as get_trend, those are command line tools which means each of them have a
    proper documentation. It suffices to simply call them in windows CMD by typing python "script_name.py" -h for it to
    print the full explanation of its use cases
    :param sensor:
    :return:
    """
    return {
        'YOC': {
            'fp': 'G:\\Data\\NW\\YOC\\{init}*.hdf',
            'op': 'C:\\Users\\Eligio\\Documents\\NPEC\\NEAT\\Data\\{sen}\\{fid}',
            'pp': f'{os.path.dirname(os.getcwd())}\\Figures\\{{sen}}\\{{fid}}',
            'if': ('_Max_trend_sen_90.nc', '_Max_trend_sen_90.nc', '_comp.nc', 'ValidCounts.nc'),
            'dv': ('Sen_slope_90', 'neat'),  # , 'Chl', '{year}_Count'
            'by': 1998,
        },
        'AQUA': {
            'fp': 'G:\\Data\\NW\\AQUA\\{init}*.hdf',
            'op': 'C:\\Users\\Eligio\\Documents\\NPEC\\NEAT\\Data\\{sen}\\{fid}',
            'pp': f'{os.path.dirname(os.getcwd())}\\Figures\\{{sen}}\\{{fid}}',
            'if': ('_Max_trend_sen_90.nc', '_Max_trend_sen_90.nc', '_comp.nc', 'ValidCounts.nc'),
            'dv': ('Sen_slope_90', 'neat'),  # , 'Chl', '{year}_Count'
            'by': 2003,
        },
        'AQUA_L3M': {
            'fp': 'G:\\Data\\Global\\MODIS-Aqua\\L3M\\{init}*.hdf',
            'op': 'C:\\Users\\Eligio\\Documents\\NPEC\\NEAT\\Data\\{sen}\\{fid}',
            'pp': f'{os.path.dirname(os.getcwd())}\\Figures\\{{sen}}\\{{fid}}',
            'if': ('_Max_trend_sen_90.nc', '_Max_trend_sen_90.nc', '_comp.nc', 'ValidCounts.nc'),
            'dv': ('Sen_slope_90', 'neat'),  # , 'Chl', '{year}_Count'
            'by': 2003,
        },
        'AQUA_L3B': {
            'fp': 'G:\\Data\\Global\\MODIS-Aqua\\L3BMax\\{init}*L3m_MAX_CHL.hdf',
            'op': 'C:\\Users\\Eligio\\Documents\\NPEC\\NEAT\\Data\\{sen}\\{fid}',
            'pp': f'{os.path.dirname(os.getcwd())}\\Figures\\{{sen}}\\{{fid}}',
            'if': ('_Max_trend_sen_90.nc', '_Max_trend_sen_90.nc', '_comp.nc', 'ValidCounts.nc'),
            'dv': ('Sen_slope_90', 'neat'),  # , 'Chl', '{year}_Count'
            'by': 2003,
        }
    }[sensor]


def neat_range(base_year: int):
    for ey in range(2015, int(time.strftime("%Y"))):
        yield f'{base_year}{ey}'


if __name__ == '__main__':
    tst = time.perf_counter()
    slope = ('CAT', 'NUM')[1]

    for sen in ('YOC', 'AQUA', 'AQUA_L3M', 'AQUA_L3B')[:1]:
        params = sen_params(sensor=sen)
        input_path = os.path.dirname(params['fp'])
        n = len(input_path) + len(f'SEN: {sen} | ')
        logger.info(f'{"-" * n}\nSEN: {sen} | {input_path}\n{"-" * n}')

        for var, f in zip(params['dv'], params['if']):
            for fid in neat_range(base_year=params['by']):
                logger.info(fid)

                out_path = params['op'].format(sen=sen, fid=fid)
                png_path = params['pp'].format(sen=sen, fid=fid)
                if os.path.isdir(png_path) is False:
                    os.makedirs(png_path)
                if os.path.isdir(out_path) is False:
                    os.makedirs(out_path)

                # Get trend
                # =======================================
                trend = f'{out_path}\\{sen[0]}{fid}*_Max_trend_sen_90.nc'
                # logger.info(f'{trend}\n{glob(trend)}')
                file_pattern = params['fp'].format(init=sen[0])

                if len(glob(trend)) == 0:
                    trend = get_trend(
                        ipat=file_pattern,
                        outpath=out_path,
                        syear=int(fid[:4]),
                        eyear=int(fid[4:])
                    )
                else:
                    trend = glob(trend)[0]

                # get comp
                # =======================================
                comp = f'{out_path}\\{sen[0]}' \
                       f'{int(fid[4:]) - 2}{int(fid[4:])}*comp.nc'
                if len(glob(comp)) == 0:
                    if 'L3B' in sen:
                        comp = l3b_comp(
                            ipat=file_pattern,
                            outpath=out_path,
                            eyear=int(fid[4:]))
                    else:
                        comp = get_composite(
                            ipat=file_pattern,
                            outpath=out_path,
                            eyear=int(fid[4:])
                        )
                else:
                    comp = glob(comp)[0]

                # get neat map
                # =======================================
                neat_file = get_neat_map(trend_file=trend,
                                         comp_file=comp)

                slope_case = 'slope' in var
                in_file = comp if var == 'Chl' else neat_file
                name = var.split('_')[1].upper() if slope_case else var.upper()
                trail = f'_{slope}' if ((slope == 'CAT') and slope_case) else ''

                # Save figures
                # =======================================
                # subareas = list(SubArea('OC').names()) + ['BS'] 'BS',
                # subareas = ['NW', 'ECS']
                subareas = ['BS']
                for sba in subareas:
                    trg_file = os.path.basename(
                        f'{in_file.split("_")[0]}_'
                        f'{name}{trail}.{sba}.png')
                    trg_file = f'{png_path}\\{trg_file}'

                    if os.path.isfile(trg_file):
                        logger.info(f'Continue\n{trg_file}\n')
                        continue
                    get_figures(sub=sba, vid=var,
                                png_file=trg_file,
                                file=in_file,
                                syear=int(fid[:4]),
                                eyear=int(fid[4:])
                                )

        time_elapsed = (time.perf_counter() - tst)
        hrs = int(time_elapsed // 3600)
        mnt = int(time_elapsed % 3600 // 60)
        sec = int(time_elapsed % 3600 % 60)
        logger.info(f'Processing Time:{hrs:3} hrs'
                    f'{mnt:3} min{sec:3} sec')
