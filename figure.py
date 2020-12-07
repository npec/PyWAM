# ----------------------------------------------------------------------------------------------------------------------
# Name:        FIGURES
# Purpose:     CLI for creating NEAT images
#
# Author:      Eligio
#
# Created:     01/11/2020
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
from NEWDAP.utils import SubArea
from logger import config
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = config(logger=logger)

__version__ = '1.0.0'


def cmap_read(cmap: str):
    import cmaps

    fid = '{}/{}'.format(cmaps.__path__[0], cmap)
    colours = list()
    with open(fid, 'r') as f:
        for line in f.readlines():
            row = line.split()
            colours.append(
                [eval(row[0]),
                 eval(row[1]),
                 eval(row[2])]
            )
            # idy = np.where(rgb == np.max(rgb))[0]
    n = np.array(colours).shape[0]
    levels = (np.array(range(n)) / float(n)).tolist()  # type: list
    levels[-1] = 1.0
    return colours, levels, n


def fix_ticks(cb):
    return [f'{(round(float(t) * 100) / 100):g}'
            for t in cb.get_ticks()]


class GetFigure:
    __slots__ = 'caxis', 'cmap', 'debug', 'file', 'lat', 'lon', 'plt', 'sds', \
                'start', 'subarea', 'threshold', 'transparent', 'trend_file', \
                'trg_file', 'var', 'verbose', 'yrange'

    def __init__(self, file: str, var: str, **kwargs):
        self.start = time.perf_counter()
        self.file = file
        self.var = var
        self.cmap = kwargs.pop('cmap').lower()
        self.trg_file = kwargs.pop('trg_file')
        self.debug = kwargs.pop('debug')
        self.plt = None

        with Dataset(self.file, 'r') as nc:
            self.sds = nc[self.var][:]
            if self.debug:
                logger.info(f'{self.file}: {self.var}\n{self.trg_file}')

            if 'slope' in self.var.lower():
                zero = 128 * nc[self.var].getncattr(
                    'scale_factor') + nc[self.var].getncattr('add_offset')
                self.sds[self.sds == zero] = 0

                if self.cmap == 'cat':
                    self.sds[np.ma.where(self.sds > 0)] = 3
                    self.sds[np.ma.where(self.sds < 0)] = 1
                    self.sds[np.ma.where(self.sds == 0)] = 2

            if ('chl' in self.var.lower()) and (self.cmap == 'cat'):
                self.threshold = kwargs.pop('threshold')
                self.sds[np.ma.where(
                    (self.sds < self.threshold))] = -1
                self.sds[np.ma.where(
                    (self.sds >= self.threshold))] = 1

            self.lon = nc['lon'][:]
            self.lat = nc['lat'][:]

            if not self.file.endswith('_comp.nc'):
                file = glob(f'{os.path.dirname(self.file)}/*_comp.nc')
                if len(file) > 0:
                    with Dataset(file[0], 'r') as mc:
                        mask = mc['Chl'][:]
                    self.sds = np.ma.masked_where(mask == 1, self.sds)

            self.caxis = kwargs.pop('caxis')

            # self.opath = kwargs.pop('opath')
            self.subarea = kwargs.pop('subarea')
            self.verbose = kwargs.pop('verbose')
            self.yrange = kwargs.pop('yrange')
            self.transparent = kwargs.pop('transparent')

    def colorbar(self):
        if 'slope' in self.var.lower():
            title = f'Trend in Annual CHL Max.\n' \
                    f'({self.yrange[0]}-{self.yrange[1]})' \
                if self.yrange is not None else 'Trend in Annual CHL Max'

            if self.cmap == 'cat':
                return {'cax': [0, 4],
                        'ticks': [.75, 2, 3.25],
                        'labels': ['D', 'N', 'I'],
                        'unit': None,
                        'title': f'{title}',
                        'nclors': 4}

            if self.cmap == 'num':
                return {'cax': [-0.35, 0.35],
                        'ticks': None,
                        'labels': None,
                        'unit': 'Trend [mg m$^{-3}$ y$^{-1}$]',
                        'title': f'{title}',
                        'nclors': 31}

        if 'chl' in self.var.lower():
            title = f'3-year Mean CHL\n' \
                    f'({self.yrange[0]}-{self.yrange[1]})' \
                if self.yrange is not None else '3-year Mean CHL'

            if self.cmap == 'cat':
                return {'cax': [-1, 1],
                        'ticks': [-1, 0, 1],
                        'labels': ['LOW', f'{self.threshold:g}', 'HIGH'],
                        'unit': None,
                        'title': f'{title}',
                        'nclors': 3}

            if self.cmap == 'num':
                return {'cax': [0.01, 64],
                        'ticks': [0.01, 0.1, 1, 10, 60],
                        'labels': None,
                        'unit': 'CHL [mg m$^{-3}$]',
                        'title': f'{title}',
                        'nclors': 256}

        if self.var == 'neat':
            return {'cax': [1, 7],
                    'ticks': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                    'labels': ['LD', 'LN', 'LI', 'HD', 'HN', 'HI'],
                    'unit': None,
                    'title': None,
                    'nclors': 7}

        if 'count' in self.var.lower():
            if self.cmap == 'cat':
                return {'cax': [0, 100],
                        'ticks': [0, 25, 50, 75, 100],
                        'labels': None,
                        'unit': '[%]',
                        'title': 'Percent Valid Pixels',
                        'nclors': 21}

            if self.cmap == 'num':
                return {'cax': [0, (self.yrange[1] - self.yrange[0]) * 12],
                        'ticks': None,
                        'labels': None,
                        'unit': 'N [Count]',
                        'title': 'Valid Pixel Number',
                        'nclors': 21}

    def custom_cmap(self, plt, cmap: str):
        """
            Reads colormap files and saves it in the current pyplot object.
            :param plt:
            :param cmap:
            """

        if (self.var in ('neat', 'Chl')) or \
                ('Sen_slope' in self.var):
            colours, levels, n = self.neat_cmap()
        else:
            colours, levels, n = cmap_read(cmap=cmap)

        stop = 0
        r, g, b = [], [], []
        while stop < n:
            r.append(tuple([levels[stop], colours[stop][0], colours[stop][0]]))
            g.append(tuple([levels[stop], colours[stop][1], colours[stop][1]]))
            b.append(tuple([levels[stop], colours[stop][2], colours[stop][2]]))
            stop += 1
        colour_map = {
            'red': tuple(r),
            'green': tuple(g),
            'blue': tuple(b)
        }

        cmap = colors.LinearSegmentedColormap('custom_cmp', colour_map)
        plt.register_cmap(cmap=cmap)
        return colour_map, plt

    def get_norm(self):
        cb = self.colorbar()
        vmin, vmax = cb['cax']
        ncl = cb['nclors']

        if (self.var == 'Chl') and \
                (self.cmap == 'num'):
            return colors.LogNorm(vmin=vmin, vmax=vmax)

        bounds = np.linspace(vmin, vmax, ncl)
        return colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    def init_fig(self, cmap: str = None):
        """
        Initialises and updates matplotlib images from the pyplot plt object.

        :param cmap:
        :return: fig, pad, im
        """
        import matplotlib
        matplotlib.use('Agg')

        import warnings
        from matplotlib import pyplot as plt

        warnings.catch_warnings()
        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)

        im = SubArea(aoi=self.subarea).bbox()  # .subarea_limits()
        if self.subarea == 'NW':
            fs, lw, pad, tpad, ms = 108, 8, 0.3, 33, 30
        elif self.subarea == 'global':
            fs, lw, pad, tpad, ms = 108, 8, 0.3, 33, 30
        else:
            fs, lw, pad, tpad, ms = 40, 4, 0.15, 15, 14
            if self.subarea == 'PB':
                fs = 38

        _, self.plt = self.custom_cmap(plt, cmap=cmap)

        self.plt.rcParams.update({'font.size': fs})
        self.plt.rcParams['axes.linewidth'] = lw

        self.plt.rcParams['xtick.major.size'] = ms
        self.plt.rcParams['xtick.major.width'] = lw
        self.plt.rcParams['xtick.minor.size'] = ms / 2
        self.plt.rcParams['xtick.minor.width'] = lw - 2

        self.plt.rcParams['ytick.major.size'] = ms
        self.plt.rcParams['ytick.major.width'] = lw
        self.plt.rcParams['ytick.minor.size'] = ms / 2
        self.plt.rcParams['ytick.minor.width'] = lw - 2

        self.plt.rcParams['axes.titlepad'] = tpad

        idy = np.where((self.lat >= im.lat_min) &
                       (self.lat <= im.lat_max))[0]
        self.lat = self.lat[idy]
        self.sds = self.sds[idy, :]

        idx = np.where((self.lon >= im.lon_min) &
                       (self.lon <= im.lon_max))[0]
        self.lon = self.lon[idx]
        self.sds = self.sds[:, idx]
        return im, lw, 'custom_cmp'

    def neat_cmap(self):
        """
        :return:
        """
        rr = gg = bb = None

        if self.var == 'neat':
            # rr = [32, 1, 0, 255, 255, 180]  # 255
            # gg = [0, 196, 255, 223, 47, 6]  # 102
            # bb = [224, 255, 2, 2, 0, 36]  # 51
            rr = [32, 1, 0, 255, 255, 190]
            gg = [0, 196, 255, 223, 175, 6]
            bb = [224, 255, 2, 2, 130, 0]

        elif 'Sen_slope' in self.var:
            if self.cmap == 'num':
                return cmap_read(cmap='anomaly.dat')
            rr = [0, 191, 190]
            gg = [24, 171, 0]
            bb = [190, 4, 24]

        elif 'chl' in self.var.lower():  # HL
            if self.cmap == 'num':
                return cmap_read(cmap='chl_cmap.dat')
            rr = [1, 250]
            gg = [196, 111]
            bb = [255, 21]

        rr, gg, bb = (np.array(rr) / 255., np.array(gg) / 255., np.array(bb) / 255.)

        colours = list()
        for r, g, b in zip(rr, gg, bb):
            colours.append([r, g, b])

        n = np.array(colours).shape[0]
        # percent level of each colour in the colour map
        levels = (np.array(range(n)) / float(n - 1)).tolist()  # type: list
        return colours, levels, n

    def proj(self, im):
        # 'gall', 'tmerc', 'moll',
        # 'omerc', 'lcc', 'moll'
        if 'glob' in self.subarea:
            return {'mlp': [0, 0, 0, 0],  # [0, 0, 1, 0],
                    'lat_ts': 0,
                    'lon_0': 0,
                    'lat_0': 0,
                    'proj': 'gall',
                    'res': 'h',
                    'plp': [False, False, False, False]  # [True, False, False, False]
                    }

        return {'mlp': [0, 0, 0, 0],  # [0, 0, 0, 1],
                'lat_ts': round((im.lat_min + im.lat_max) / 2),
                'proj': 'tmerc',
                'lat_0': round((im.lat_min + im.lat_max) / 2),
                'lon_0': round((im.lon_min + im.lon_max) / 2),
                'res': 'f',
                'plp': [False, False, False, False]
                }

    def save(self):
        from mpl_toolkits.basemap import Basemap

        loc, pos = ('bottom', 'horizontal') \
            if 'glob' in self.subarea else ('right', 'vertical')
        im, lw, cmp = self.init_fig(cmap=self.cmap)
        fig, ax = self.plt.subplots(figsize=(im.width, im.height), dpi=im.dpi)

        mp = self.proj(im=im)
        m = Basemap(projection=mp.pop('proj'),
                    lat_ts=mp.pop('lat_ts'),
                    lat_0=mp.pop('lat_0'),
                    lon_0=mp.pop('lon_0'),
                    resolution=mp.pop('res'),
                    llcrnrlat=im.lat_min,
                    urcrnrlat=im.lat_max,
                    llcrnrlon=im.lon_min,
                    urcrnrlon=im.lon_max)

        m.drawmeridians(np.arange(im.xlabel0, im.xlabel1, im.x_step),
                        labels=mp['mlp'], linewidth=lw)
        m.drawparallels(np.arange(im.ylabel0, im.ylabel1, im.y_step),
                        labels=mp['plp'], linewidth=lw)

        if len(self.lon.shape) == 1:
            lon, lat = m(*np.meshgrid(self.lon, self.lat))
        else:
            lon, lat = m(self.lon, self.lat)

        # trg_file = os.path.basename(
        #     f'{self.file.split("_")[0]}_'
        #     f'{self.var[:6].upper()}.{self.subarea}.png')

        if 'glob' in self.subarea:
            m.drawlsmask(land_color=[.8, .8, .8], ocean_color='black', resolution='h')
        else:
            m.fillcontinents(color=[.8, .8, .8], lake_color='black')  # Fill the continents
        if self.transparent is True:
            m.fillcontinents(color=[.8, .8, .8], lake_color='white', alpha=0)  # Fill the continents
        m.drawmapboundary(linewidth=lw, fill_color='black')  # Fill the globe with a blue color
        m.drawcoastlines(linewidth=1)
        # m.drawlsmask(ocean_color='black', resolution='f')
        # m.drawrivers(linewidth=3, linestyle='solid', color='b', antialiased=1, ax=None, zorder=None)

        mesh = m.pcolormesh(lon, lat, self.sds,
                            shading='flat',
                            norm=self.get_norm(),
                            cmap=self.plt.get_cmap(name=cmp))
        # , lut=cbp['nclors']
        # plt.show()
        cbp = self.colorbar()
        if cbp['title'] is not None:
            self.plt.title(cbp['title'])

        divider = make_axes_locatable(axes=ax)
        cax = divider.append_axes(loc, size="2%", pad=0.2)
        cb = self.plt.colorbar(mesh, cax=cax, orientation=pos, format='%g')

        if cbp['ticks'] is not None:
            cb.set_ticks(cbp['ticks'])
        if cbp['labels'] is not None:
            cb.set_ticklabels(cbp['labels'])
        if cbp['unit'] is not None:
            cb.set_label(cbp['unit'])

        if ('slope' in self.var) and \
                (self.cmap == 'num'):
            cb.set_ticklabels(fix_ticks(cb=cb))

        if ('chl' in self.var.lower()) and \
                (self.cmap == 'num'):
            cb.ax.minorticks_on()

        self.plt.tight_layout()
        opath = os.path.dirname(self.trg_file)
        if not os.path.isdir(opath):
            os.makedirs(opath)
        fig.savefig(self.trg_file,  # f'{self.opath}{os.sep}{trg_file}'
                    dpi=im.dpi, bbox_inches='tight')
        self.plt.close('all')

        time_elapsed = (time.perf_counter() - self.start)
        hrs = int(time_elapsed // 3600)
        mnt = int(time_elapsed % 3600 // 60)
        sec = int(time_elapsed % 3600 % 60)
        logger.info(f'Save: {self.trg_file}\n'
                    f'ElapsedTime:{hrs:3} hrs'
                    f'{mnt:3} min{sec:3} sec\n')


def cli_main():
    """Command line parser interface for automated WAM_ANNUAL_MAX and WAM_TREND CLI.
    """
    start = time.perf_counter()
    default_dir = f'{os.path.dirname(os.getcwd())}{os.sep}Figures'
    parser = argparse.ArgumentParser(description=f'NEAT Figures ver. {__version__}'
                                                 f'\n', add_help=True,
                                     epilog=textwrap.dedent('''\
                                     Get the NEAT map from the computation of 
                                     trends (Sen's slope trend) and temporal CHL mean
                                     Type figure --help for details.
                                     --------
                                     Examples:
                                        figure trend.nc composite.nc -o ODIR
                                        '''
                                                            ),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     )
    parser.add_argument('file', help='input filename', type=str, metavar='filename')
    parser.add_argument('var', help='input file variable name', type=str, metavar='varname')
    parser.add_argument("-v", "--version", action="version", version="%(prog)s " + __version__)
    parser.add_argument('-o', help=f"output path (default: {default_dir}')",
                        dest='opath', type=str, default=default_dir)
    parser.add_argument("-s", help="subarea <NW | TB | ...> (default: NW)",
                        type=str, dest="subarea", default='NW')
    parser.add_argument("-c", help="colour map for slope or composite <NUM | CAT> with "
                                   "NUM=numerical\n (actual values) or CAT=categorical "
                                   "(default: CAT). Ignored for NEAT and other maps",
                        type=str, dest="cmap", default='CAT')
    parser.add_argument("-l", help="colour map limits for NUM <min max> "
                                   "(default: from data min/max). Ignored for NEAT",
                        type=list, dest="caxis", nargs=2, default=None)
    parser.add_argument("-y", help="year range to include in the figure title <start end>",
                        type=list, dest="yrange", nargs=2, default=None)
    parser.add_argument("-t", help="threshold for the CAT map of mean CHL (default: 5 mg m^-3)",
                        type=float, dest="threshold", default=5)
    parser.add_argument("-b", help="image background <0 | 1> 0==transparent (default: 1)",
                        type=int, dest="transparent", default=1)

    opts = parser.parse_args()
    nf = GetFigure(
        file=opts.file,
        var=opts.var, **{'opath': opts.opath,
                         'threshold': opts.threshold,
                         'caxis': opts.caxis,
                         'subarea': opts.subarea,
                         'var': opts.var,
                         'cmap': opts.cmap,
                         'yrange': opts.yrange,
                         'transparent': not bool(opts.transparent),
                         'verbose': bool(opts.verbose),
                         'debug': False
                         })
    nf.save()

    time_elapsed = (time.perf_counter() - start)
    hrs = int(time_elapsed // 3600)
    mnt = int(time_elapsed % 3600 // 60)
    sec = int(time_elapsed % 3600 % 60)
    logger.info(f'Processing Time:{hrs:3} hrs'
                f'{mnt:3} min.{sec:3} sec.')


if __name__ == '__main__':
    cli_main()
