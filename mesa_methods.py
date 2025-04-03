import os

import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

from ipywidgets import interact, IntSlider

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker


import seaborn as sns

import scipy.signal
import scipy.integrate
from scipy.interpolate import make_smoothing_spline

from datetime import datetime
from tomso import gyre

from tqdm import tqdm

print(f'Updated module {datetime.now()}')

plt.rcParams.update({'axes.linewidth' : 1,
                     'ytick.major.width' : 1,
                     'ytick.minor.width' : 1,
                     'xtick.major.width' : 1,
                     'xtick.minor.width' : 1,
                     'xtick.labelsize': 10, 
                     'ytick.labelsize': 10,
                     'axes.labelsize': 12,
                     'font.family': 'Serif',
                      'figure.figsize': (6, 4)
                    })

 

def find_peak(x, y):
    # return x[np.argmax(y)]
    return x[scipy.signal.argrelmax(y)[0]]

types = ['X', 'O', 'B', 'A', 'F', 'G', 'K', 'M']
spectrals = np.array([1e99, 30000, 10000, 7500, 6000, 5200, 3700, 2400])
rgbs = [(1, 1, 1), # X, temp class just for setting upper bound 
        (175/255, 201/255, 1),       # O
        (199/255, 216/255, 1),       # B
        (1,       244/255, 243/255), # A 
        (1,       229/255, 207/255), # F 
        (1,       217/255, 178/255), # G 
        (1,       199/255, 142/255), # K 
        (1,       166/255, 81/255)]  # M

prof_cols = ['zone', 'mass', 'logR', 'logT', 'logRho', 'logP', 'x_mass_fraction_H', 'y_mass_fraction_He', 'z_mass_fraction_metals', 'pp', 'cno', 'tri_alpha', 'brunt_N', 'lamb_S']

def plot_colors(y, xlim=[30000, 0], ax=None):
    if ax is None:
        ax = plt.gca()

    ax.fill_betweenx(y, spectrals[0], spectrals[1], color=rgbs[1])
    ax.fill_betweenx(y, spectrals[1], spectrals[2], color=rgbs[2])
    ax.fill_betweenx(y, spectrals[2], spectrals[3], color=rgbs[3])
    ax.fill_betweenx(y, spectrals[3], spectrals[4], color=rgbs[4])
    ax.fill_betweenx(y, spectrals[4], spectrals[5], color=rgbs[5])
    ax.fill_betweenx(y, spectrals[5], spectrals[6], color=rgbs[6])
    ax.fill_betweenx(y, spectrals[6], spectrals[7], color=rgbs[7])
    # ax.fill_between(y, spectrals[7], 1000, color=rgbs[7])





class model:
    def __init__(self, logs_dir, gyre_ran=False, gyre_limit=None, gyre_dir='', load_all_in_prof=False, nonadiabatic_dir=None):
        print(f'Creating model from directory {logs_dir}')

        self.logs_dir = logs_dir

        def load_history_file():
            return pd.read_table(os.path.join(self.logs_dir, 'history.data'), 
                skiprows=5, sep=r'\s+')

        def get_index():
            return pd.read_table(os.path.join(self.logs_dir, 'profiles.index'), 
                names=['model_number', 'priority', 'profile_number'],
                skiprows=1, sep=r'\s+')
        
        self.DF = load_history_file()
        self.index = get_index()  
        
        def load_profile(profile_number):
            if load_all_in_prof is True:
                prof = pd.read_table(
                    os.path.join(self.logs_dir, 'profile' + str(profile_number) + '.data'), 
                    skiprows=5, sep=r'\s+', engine='c')
            if load_all_in_prof is False:
                prof = pd.read_table(
                    os.path.join(self.logs_dir, 'profile' + str(profile_number) + '.data'), 
                    skiprows=5, sep=r'\s+', engine='c', usecols=prof_cols)
            return prof

        def get_profiles():
            return [load_profile(profile_number) 
                    for profile_number in tqdm(self.index.profile_number)]

        print('Loading Profiles: ')
        self.profs = get_profiles()

        # def find_profile_where(profile, value, quantity):
        #     prof = load_profile(profile)
        #     return profile

        def get_frequencies(profile_number, dir=gyre_dir):
            path = os.path.join(self.logs_dir, dir, 'profile' + str(profile_number)  + '-freqs.dat')
            return pd.read_table(path, skiprows=5, sep=r'\s+')
    
        def get_all_frequencies():
            return [get_frequencies(profile_number)
                    for profile_number in self.index.profile_number]

        def is_unstable(profile_num):
            instability = True in (f > 0 for f in get_frequencies(profile_num, dir=nonadiabatic_dir)['Im(freq)'])
            return instability


        if gyre_ran is True:
            if gyre_limit is not None:
                index_profiles = np.arange(1, gyre_limit)
            if gyre_limit is None:
                # self.freqs = get_all_frequencies()
                index_profiles = self.index.profile_number
            
            print('Loading Frequencies')
            self.freqs = [get_frequencies(profile_number) for profile_number in tqdm(index_profiles)]
            
            if nonadiabatic_dir is not None:
                self.nad_freqs =  [get_frequencies(profile_number, dir=nonadiabatic_dir) for profile_number in tqdm(index_profiles)]
                
                try:
                    unstable_prof_idx = [is_unstable(prof) for prof in index_profiles]
                    self.unstable_profs = index_profiles[unstable_prof_idx]
                except:
                    print('no unstable profs ?')

            # # [float(10**cm.get_history(prof_num)['log_Teff']) for prof_num in cm.unstable_profs]
            # # hist_ = 10**self.get_history(prof_num)
            # self.unstable_temps = []
            # self.unstable_ages = []
            # for prof_num in self.unstable_profs:
            #     hist_ = self.get_history(prof_num)
            #     self.unstable_temps.append(float(10**hist_['log_Teff']))
            #     self.unstable_ages.append(float(hist_['star_age']))
            

        

        def load_gyre(profile_number):
            prof = gyre.load_gyre(
                os.path.join(self.logs_dir, 'profile' + str(profile_number) + '.data.GYRE'))
            return prof
        
        def get_gyres():
            return [load_gyre(profile_number) 
                    for profile_number in tqdm(self.index.profile_number)]
        
        print('Loading gyre files: ')
        self.gyres = get_gyres()

        # df = self.DF[50:]
        # mn = df[self.DF['mass_conv_core'][50:]==0].iloc[0].model_number
        # self.he_burn_end = self.index.profile_number[nv_ledoux.index.model_number==mn]


    def load_gyre(self, profile_number):
        prof = gyre.load_gyre(
            os.path.join(self.logs_dir, 'profile' + str(profile_number) + '.data.GYRE'))
        return prof


    def get_he_burn_end(self):
        df = self.DF[50:]
        mn = df[self.DF['mass_conv_core'][50:]==0].iloc[0].model_number
        self.he_burn_end = self.index.profile_number[self.index.model_number==mn]

        return self.he_burn_end.iloc[0]


    def get_value_profile(self, value, prof_num):
        model_num = self.index.iloc[np.argwhere(self.index.profile_number==prof_num)[0][0]].model_number
        df_mod = self.DF.iloc[np.argwhere(self.DF.model_number==model_num)[0][0]]
        return df_mod[value]
    
    
    def get_Pg(self, prof_num):
        # this is in seconds i think


        # prof = self.profs[prof_num-1]
      
        # x = 10**prof.logR / np.max(10**prof.logR)
        # N = prof.brunt_N.values/(2*np.pi)*1e6
        # delta_Pg = - 2*np.pi**2/np.sqrt(2)/ sp.integrate.trapz(N[N>0]/x[N>0], x[N>0])

        # gyre = self.load_gyre(prof_num)
        gyre = self.gyres[prof_num-1]
        N  = gyre.N
        x  = gyre.x
        delta_Pg = 2*np.pi**2/np.sqrt(2)/ sp.integrate.trapezoid(N[N>0]/x[N>0], x[N>0])

        # delta_Pg_dict = {} 
        # gyres = self.get_gyres()
        # for i, gyre in enumerate(gyres):
        #     profile_number = self.index.profile_number[i]
        #     model_number = self.index.loc[self.index.profile_number == profile_number, 'model_number'].values[0]
        #     N  = gyre.N
        #     x  = gyre.r
        #     m  = gyre.m
        #     delta_Pg = 2*np.pi**2/np.sqrt(2)/ sp.integrate.trapz(N[N>0]/x[N>0], x[N>0])
        #     delta_Pg_dict[model_number] = delta_Pg

        return delta_Pg

    
    def get_history(self, profile_number):
        model_number = self.index[self.index.profile_number == profile_number].model_number.values[0]
        return self.DF[self.DF.model_number == model_number]
    
    
        

    def plot_HR(self, ax=None, title='', profile_number=-1, mark_profs=False, mark_unstable=False, linestyle='solid', color=None, c=None, alpha=1, label=None):
        if color is None:
            color='k'
        
        if ax is None:
            ax = plt.gca()

        ax.plot(10**self.DF['log_Teff'][1:], 
                10**self.DF['log_L'][1:], 
                linestyle=linestyle,
                lw=2, c=color, alpha=alpha, label=label)
    
        first = 1
        if mark_profs:
            for prof_num in self.index.profile_number:
                hist = self.get_history(prof_num)
                ax.plot(10**hist['log_Teff'], 10**hist['log_L'], '.',
                        c=c if prof_num == profile_number else c, 
                        label=r'%0.2f Gyr' % (hist.star_age.values[0]/1e9)
                            if prof_num == profile_number else 'profile files' if first else '',
                        ms=20)
                if not prof_num == profile_number:
                    first = 0
        if mark_unstable:
            for prof_num in self.unstable_profs:
                hist = self.get_history(prof_num)
                ax.plot(10**hist['log_Teff'], 10**hist['log_L'], '.',
                        c=c if prof_num == profile_number else c, 
                        label=r'%0.2f Gyr' % (hist.star_age.values[0]/1e9)
                            if prof_num == profile_number else '',
                        ms=20)
                if not prof_num == profile_number:
                    first = 0
    
        plt.xlabel(r'Effective Temperature $T_{\rm{eff}}$ (K)')
        plt.ylabel(r'Luminosity ($\rm{L}_\odot$)')
        ax.set_yscale('log')
        ax.invert_xaxis()
    
        ax.set_title(title)
        # plt.show()



    def plot_composition(self, profile_number):
        ZAMS_X = self.profs[0].x_mass_fraction_H.values[0]
        ZAMS_Y = self.profs[0].y_mass_fraction_He.values[0]
        Y_p = 0.2463

        prof = self.profs[profile_number-1]
        x = 10**prof.logR / np.max(10**prof.logR)
        plt.plot(x, prof.x_mass_fraction_H, lw=5, label='hydrogen', c='k')
        plt.plot(x, prof.y_mass_fraction_He, lw=5, label='helium', c='b')
        plt.axhline(ZAMS_X, c='k', ls='--', zorder=-99)
        plt.axhline(ZAMS_Y, c='k', ls='--', zorder=-99)
        plt.axhline(Y_p, c='lightgray', ls='--', zorder=-99)
        plt.xlabel(r'fractional radius $r/R$')
        plt.ylabel(r'mass fraction')
        plt.legend()
        plt.title('Internal Composition', size=24)
        plt.show()

    def plot_composition_mass(self, profile_number, legend=False, legendloc=None, fillconv=False, ax=None, title=None):
        if ax is None:
            ax = plt.gca()

        ZAMS_X = self.profs[0].x_mass_fraction_H.values[0]
        ZAMS_Y = self.profs[0].y_mass_fraction_He.values[0]
        # Y_p = 0.2463

        prof = self.profs[profile_number-1]
        x = prof.mass
        

        if fillconv is True:
            ax.fill_betweenx(np.linspace(0,1), 0, self.DF['mass_conv_core'][profile_number], color='lightgray', zorder=-9999)
            ax.fill_betweenx(np.linspace(0,1), self.DF['conv_mx1_bot'][profile_number]*max(x), self.DF['conv_mx1_top'][profile_number]*max(x), color='lightgray', zorder=-9999)
            ax.fill_betweenx(np.linspace(0,1), self.DF['conv_mx2_bot'][profile_number]*max(x), self.DF['conv_mx2_top'][profile_number]*max(x), color='lightgray', zorder=-9999)


        ax.plot(x, prof.x_mass_fraction_H, lw=3, label='H', c='k')
        ax.plot(x, prof.y_mass_fraction_He, lw=3, label='He', c='green')
        # plt.axhline(ZAMS_X, c='k', ls='--', zorder=-99)
        # plt.axhline(ZAMS_Y, c='k', ls='--', zorder=-99)
        # plt.axhline(Y_p, c='lightgray', ls='--', zorder=-99)
        # plt.axvline(self.DF['mass_conv_core'][profile_number], ls='--', c='k', lw=1)

        

        ppcnomin = np.min(prof.mass[(prof.pp+prof.cno) > 0.001])
        ppcnomax = np.max(prof.mass[(prof.pp+prof.cno) > 0.001]) 
        
        try:
            triamin = np.min(prof.mass[(prof.tri_alpha) > 0.001]) 
            triamax = np.max(prof.mass[(prof.tri_alpha) > 0.001])
        except:
            triamin = np.min(prof.mass[(prof.tri_alfa) > 0.001]) 
            triamax = np.max(prof.mass[(prof.tri_alfa) > 0.001])


    
        ax.fill_betweenx(np.linspace(0, 1), ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.5, lw=0, zorder=-999)
        ax.fill_betweenx(np.linspace(0, 1), triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=0.5,   lw=0, zorder=-999)

        ax.set_xlim(0,max(x))
        ax.set_ylim(0,1)
        ax.set_xlabel(r'Fractional Mass $m/M_\odot$')
        ax.set_ylabel(r'Mass Fraction')
        if legend is True:
            ax.legend(loc=legendloc)
        if title is not None:
            ax.set_title(title)
        # plt.title('Internal Composition', size=24)
        # plt.show()
    
    
    def plot_propagation(self, profile_number, ax=None, ell=1, mass=False):
        hist = self.get_history(profile_number)
        prof = self.profs[profile_number-1]

        if ax is None:
            ax = plt.gca()
    
        x = 10**prof.logR / np.max(10**prof.logR)
        if mass is True:
            x = prof.mass 

        brunt = prof.brunt_N.values/(2*np.pi)*1e6
        lamb  = prof.lamb_S.values*1e6/np.sqrt(2)*np.sqrt(ell*(ell+1))
        ax.plot(x, brunt, lw=3, label='Buoyancy')
        ax.plot(x, lamb, lw=3, label='Lamb')
    
        gmodes = np.minimum(brunt, lamb)
        pmodes = np.maximum(brunt, lamb)
        ax.fill_between(x, 
                        np.zeros(len(gmodes)), 
                        gmodes, 
                        color='blue', alpha=0.1, zorder=-99)
        ax.fill_between(x, 
                        1e99*np.ones(len(pmodes)), 
                        pmodes, 
                        color='orange', alpha=0.1, zorder=-99)
    
        nu_max   = hist.nu_max.values[0]
        Delta_nu = hist.delta_nu.values[0]
        # plt.axhline(nu_max, ls='--', c='k', label=r'$\nu_\max$', zorder=100)
        # plt.fill_between([0, 1], 
        #                 nu_max-5*Delta_nu, 
        #                 nu_max+5*Delta_nu, 
                        # color='#aaaaaa', zorder=-98)
    
        N = brunt
        delta_Pg = - 2*np.pi**2/np.sqrt(2)/ sp.integrate.trapezoid(N[N>0]/x[N>0], x[N>0])
        # plt.text(x=0.1, y=10**6, s='delta Pg = %.5f' % delta_Pg)
    
        ax.semilogy()
        ax.set_ylim([1, 1e6]) #500*nu_max])
        ax.set_xlim([0,1])
        ax.set_ylabel(r'Frequency $\nu/\mu\rm{Hz}$')
        ax.set_xlabel(r'Fractional Radius $r/R$')
        if mass is True:
            ax.set_xlabel(r'Fractional Mass $m/M_\odot$')
            ax.set_xlim([0, max(prof.mass)])
        ax.legend()
        # plt.title(f'Propagation Diagram {profile_number}', size=24)
        # plt.show()

    def plot_propagation_period(self, profile_number, ell=1):
        hist = self.get_history(profile_number)
        prof = self.profs[profile_number-1]
    
        x = 10**prof.logR / np.max(10**prof.logR)
        brunt = prof.brunt_N.values/(2*np.pi)*1e6
        lamb  = prof.lamb_S.values*1e6/np.sqrt(2)*np.sqrt(ell*(ell+1))
        plt.plot(x, 1/(brunt * 10**-6), lw=3, label='Buoyancy')
        # plt.plot(x, 1/(lamb*10**-6), lw=3, label='Lamb')
    
        gmodes = np.minimum(brunt, lamb)
        # pmodes = np.maximum(brunt, lamb)
        plt.fill_between(x, 
                        np.zeros(len(gmodes)), 
                        gmodes, 
                        color='blue', alpha=0.1, zorder=-99)
        # plt.fill_between(x, 
        #                 1e99*np.ones(len(pmodes)), 
        #                 pmodes, 
        #                 color='orange', alpha=0.1, zorder=-99)
    
        
    
        plt.semilogy()
        plt.ylim([1, 1e7]) #500*nu_max])
        plt.xlim([0,1])
        plt.ylabel(r'Period (s)')
        plt.xlabel(r'fractional radius $r/R$')
        plt.legend()
        # plt.title(f'Propagation Diagram {profile_number}', size=24)
        # plt.show()

    
    def get_propagation(self, profile_number, ell=1):
        hist = self.get_history(profile_number)
        prof = self.profs[profile_number-1]
    
        x = 10**prof.logR / np.max(10**prof.logR)
        brunt = prof.brunt_N.values/(2*np.pi)*1e6

        return [x, brunt]


    def get_boundaries(self, profile_number):
        x, N = self.get_propagation(profile_number)
        return find_peak(x, N)
    

    def plot_nablas(self, prof, ax=None, colors=None, plot_nabla_diff=False, plot_brunt=False, ylim1=[0, 0.5], xlim=[0,1], ylim2=[0,1], legend=True):
        gyre = self.load_gyre(prof)

        # colors = ['tomato', 'goldenrod', 'mediumseagreen', 'cornflowerblue', 'mediumpurple', 'orchid']
        if colors is None:
            colors = ['#264653', '#287271', '#2A9D8F','#E9C46A', '#F4A261', '#E76F51']



        if ax is None:
            ax = plt.gca()

        # ax.plot(gyre.m/gyre.M, gyre.nabla, color=colors[0], label='∇', zorder=999)
        ax.plot(gyre.m/gyre.M, gyre.grad_a, color=colors[1], label=r'∇$_\mathrm{ad}$')
        ax.plot(gyre.m/gyre.M, gyre.grad_r, color=colors[2], label=r'∇$_\mathrm{rad}$')
        # ax.plot(gyre.m/gyre.M, gyre.grad_r - gyre.AA, color=colors[3], label=r'∇$_\mathrm{L}$')

        # y  = nabla_rad - nabla_L
        # ax.plot(gyre.m/gyre.M, gyre.grad_L)
        
        if plot_nabla_diff is True:
            ax.plot(gyre.m/gyre.M, gyre.nabla-gyre.grad_a, color=colors[3], label=r'∇-∇$_{ad}$')
        if plot_brunt is True:
            ax.plot(gyre.m/gyre.M, gyre.N, color=colors[4], label='N')

        ax.set_ylim(ylim1)
        ax.set_xlim(xlim)

        # ax.axvline(x=self.get_value_profile('mass_conv_core', prof), color='black', label='Mass Conv Core')

        # ax1= ax.twinx()
        # ax.plot(gyre.m/gyre.M, gyre.kappa, color=colors[5], label='Opacity')
        # ax.set_ylim(ylim2)

        # lines, labels = ax.get_legend_handles_labels()
        # lines2, labels2 = ax.get_legend_handles_labels()
        # ax.legend(lines + lines2, labels + labels2, loc=(1.2, 0.4))

        # ybox1 = TextArea(r'∇$_\mathrm{ad}$', textprops=dict(color=colors[1], size=15,rotation=90,ha='left',va='bottom'))
        # ybox2 = TextArea(r'∇$_\mathrm{rad}$',     textprops=dict(color=colors[2], size=15,rotation=90,ha='left',va='bottom'))

        # ybox = VPacker(children=[ybox1, ybox2],align="bottom", pad=0, sep=5)

        # anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.08, 0.4), 
        #                                 bbox_transform=ax.transAxes, borderpad=0.)

        # ax.add_artist(anchored_ybox)

        if legend is True:
            ax.legend()

        ax.set_xlabel('Fractional mass m/M')
        ax.set_ylabel('Temperature gradient')
        # ax1.set_ylabel(r'Opacity $\kappa$')

    

    def plot_kippenhahn(self, xlim=None, xlabel='Age [Gyr]', alpha=True, xm=np.linspace(0, 17, 10000), mdl=None, ax=None, title=None, show_colorbar=True, cmap_color=mpl.cm.Greens):
        mass_max = np.min(np.array([np.max(prof.mass) for prof in self.profs]))
        print(mass_max)
        xm = np.linspace(0, mass_max, 10000)

        ages = np.array([self.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in self.index.profile_number])
        
        X, Y = np.meshgrid(xm, ages)
        Z = np.array([sp.interpolate.interp1d(p.mass, np.log10(p.brunt_N/(2*np.pi)), 
                                              fill_value=np.nan, bounds_error=0)(xm) 
                                              for p in self.profs])
        
        conv = np.array([sp.interpolate.interp1d(p.mass, p.brunt_N<0, 
                    fill_value=np.nan, bounds_error=0)(xm) 
                for p in self.profs])


        ppcnomin = np.array([np.min(p.mass[(p.pp+p.cno) > 0.001]) for p in self.profs])
        ppcnomax = np.array([np.max(p.mass[(p.pp+p.cno) > 0.001]) for p in self.profs])
        
        try:
            triamin = np.array([np.min(p.mass[(p.tri_alpha) > 0.001]) for p in self.profs])
            triamax = np.array([np.max(p.mass[(p.tri_alpha) > 0.001]) for p in self.profs])
        except:
            triamin = np.array([np.min(p.mass[(p.tri_alfa) > 0.001]) for p in self.profs])
            triamax = np.array([np.max(p.mass[(p.tri_alfa) > 0.001]) for p in self.profs])
    
        plt.figure(figsize=(7,6.5))

        if ax is None:
            ax = plt.gca()
    
        norm = mpl.colors.Normalize(vmin=-6, vmax=0)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_color)
        vmin = int(norm.vmin)
        vmax = int(norm.vmax)
    
        ax.contourf(Y, X, conv, levels=[0,1,2,3,4,5,6,7], vmin=-1, vmax=3, cmap='Greys', zorder=-99999)
        ax.contourf(Y, X, Z, levels=np.arange(-6, 1, 1), vmin=vmin, vmax=vmax, cmap=cmap_color, zorder=-99999)

        ax.set_rasterization_zorder(-1)
    
        ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
        ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)


    
        try:
            specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF.log_Teff])
            for cidx in np.unique(specidx):
                idx = specidx == cidx
                for ii, x in enumerate(idx):
                    if not x and ii+1<len(idx) and idx[ii+1]:
                        idx[ii] = 1
            
                ax.plot(ages[idx], self.DF.star_mass[idx], c='w', lw=10, zorder=-999)
                ax.plot(ages[idx], self.DF.star_mass[idx], c=rgbs[cidx], lw=8, zorder=-99)
        except:
            print('only 100 profiles')
            specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF[len(self.DF)-100:len(self.DF)].log_Teff])
            for cidx in np.unique(specidx):
                idx = specidx == cidx
                for ii, x in enumerate(idx):
                    if not x and ii+1<len(idx) and idx[ii+1]:
                        idx[ii] = 1
                ax.plot(ages[idx], self.DF[len(self.DF)-100:len(self.DF)].star_mass[idx], c='w', lw=10, zorder=-999)
                ax.plot(ages[idx], self.DF[len(self.DF)-100:len(self.DF)].star_mass[idx], c=rgbs[cidx], lw=8, zorder=-99)
    
    
        # radii = [[prof.mass[np.argmin((10**prof.logR - radius)**2)] 
        #         if radius<10**prof.logR[0] else np.nan 
        #         for prof in self.profs]
        #         for radius in [0.01, 0.05, 0.1, 0.5, 1, 10]]#, 100]]
        # for rad in radii:
        #     ax.plot(ages, rad, ls='-', c='k', lw=1, zorder=-99)


        if mdl is not None:
            #plt.axvline(track.get_history(mdl).star_age.values[0]/1e6, ls='--', c='gray')
            ax.plot(self.get_history(mdl).star_age.values[0]/1e9, 
                    self.get_history(mdl).star_mass.values[0]+1, marker='*', ms=20, c='k')
    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'Fractional mass $\mathrm{m/M_\odot}$')
        ax.set_ylim(0, mass_max * 1.1)
        #plt.gca().set_yticks([0, 5, 10, 15])
    
        if xlim is not None:
            ax.set_xlim(xlim)
    
        if show_colorbar is True:
            cb = plt.colorbar(cmap, label=r'Buoyancy frequency $\mathrm{log~N}$/Hz',
                        boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        ticks=np.array(range(vmin, vmax+1, 1)),
                        ax=ax)
            cb.ax.minorticks_off()

        ax.set_title(title)
    
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.151)

        #plt.show()



    def plot_kippenhahn_conv(self, xlim=None, xlabel='Age [Gyr]', alpha=True, xm=np.linspace(0, 17, 10000), mdl=None, ax=None, title=None, show_colorbar=True):
        mass_max = np.min(np.array([np.max(prof.mass) for prof in self.profs]))
        print(mass_max)
        xm = np.linspace(0, mass_max, 10000)

        ages = np.array([self.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in self.index.profile_number])
        
        X, Y = np.meshgrid(xm, ages)
        # Z = np.array([sp.interpolate.interp1d(p.mass, np.log10(p.brunt_N/(2*np.pi)), 
        #                                       fill_value=np.nan, bounds_error=0)(xm) 
        #                                       for p in self.profs])
    
        # conv = np.array([sp.interpolate.interp1d(p.mass, p.brunt_N<0, 
        #             fill_value=np.nan, bounds_error=0)(xm) 
        #         for p in self.profs])
        


        ppcnomin = np.array([np.min(p.mass[(p.pp+p.cno) > 0.001]) for p in self.profs])
        ppcnomax = np.array([np.max(p.mass[(p.pp+p.cno) > 0.001]) for p in self.profs])
        
        try:
            triamin = np.array([np.min(p.mass[(p.tri_alpha) > 0.001]) for p in self.profs])
            triamax = np.array([np.max(p.mass[(p.tri_alpha) > 0.001]) for p in self.profs])
        except:
            triamin = np.array([np.min(p.mass[(p.tri_alfa) > 0.001]) for p in self.profs])
            triamax = np.array([np.max(p.mass[(p.tri_alfa) > 0.001]) for p in self.profs])
    
    
        plt.figure(figsize=(7,6.5))

        if ax is None:
            ax = plt.gca()
    
        norm = mpl.colors.Normalize(vmin=-6, vmax=0)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
        vmin = int(norm.vmin)
        vmax = int(norm.vmax)
    
        ax.fill_between(ages, 0, self.DF['mass_conv_core'])
        # ax.contourf(Y, X, conv, levels=[0,1,2,3,4,5,6,7], vmin=-1, vmax=3, cmap='Greys', zorder=-99999)
        # ax.contourf(Y, X, Z, levels=np.arange(-6, 1, 1), vmin=vmin, vmax=vmax, cmap='Greens', zorder=-99999)

        ax.set_rasterization_zorder(-1)
    
        ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
        ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)
    
        try:
            specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF.log_Teff])
            for cidx in np.unique(specidx):
                idx = specidx == cidx
                for ii, x in enumerate(idx):
                    if not x and ii+1<len(idx) and idx[ii+1]:
                        idx[ii] = 1
            
                ax.plot(ages[idx], self.DF.star_mass[idx], c='w', lw=10, zorder=-999)
                ax.plot(ages[idx], self.DF.star_mass[idx], c=rgbs[cidx], lw=8, zorder=-99)
        except:
            print('only 100 profiles')
            specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF[len(self.DF)-100:len(self.DF)].log_Teff])
            for cidx in np.unique(specidx):
                idx = specidx == cidx
                for ii, x in enumerate(idx):
                    if not x and ii+1<len(idx) and idx[ii+1]:
                        idx[ii] = 1
                ax.plot(ages[idx], self.DF[len(self.DF)-100:len(self.DF)].star_mass[idx], c='w', lw=10, zorder=-999)
                ax.plot(ages[idx], self.DF[len(self.DF)-100:len(self.DF)].star_mass[idx], c=rgbs[cidx], lw=8, zorder=-99)
    
    
        # radii = [[prof.mass[np.argmin((10**prof.logR - radius)**2)] 
        #         if radius<10**prof.logR[0] else np.nan 
        #         for prof in self.profs]
        #         for radius in [0.01, 0.05, 0.1, 0.5, 1, 10]]#, 100]]
        # for rad in radii:
        #     ax.plot(ages, rad, ls='-', c='k', lw=1, zorder=-99)

    
        if mdl is not None:
            #plt.axvline(track.get_history(mdl).star_age.values[0]/1e6, ls='--', c='gray')
            ax.plot(self.get_history(mdl).star_age.values[0]/1e9, 
                    self.get_history(mdl).star_mass.values[0]+1, marker='*', ms=20, c='k')
    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'Fractional mass $\mathbf{m/M_\odot}$')
        ax.set_ylim(0, mass_max * 1.1)
        #plt.gca().set_yticks([0, 5, 10, 15])
    
        if xlim is not None:
            ax.set_xlim(xlim)
    
        if show_colorbar is True:
            cb = plt.colorbar(cmap, label=r'Buoyancy frequency $\mathbf{log~N}$/Hz',
                        boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        ticks=np.array(range(vmin, vmax+1, 1)),
                        ax=ax)
            cb.ax.minorticks_off()

        ax.set_title(title)
    
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.151)

        #plt.show()



    def plot_kippenhahn_radius(self, xlim=None, xlabel='Age [Gyr]', mdl=None, ax=None, title=None, show_colorbar=True, nolabels=False, lower_y=0.005):
        r_min = np.min(np.array([np.min(10**prof.logR) for prof in self.profs]))
        r_max = np.max(np.array([np.max(10**prof.logR) for prof in self.profs]))
        xr = np.linspace(r_min, r_max, 10000)

        ages = np.array([self.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in self.index.profile_number])
        
        X, Y = np.meshgrid(xr, ages)
        Z = np.array([sp.interpolate.interp1d(10**p.logR, np.log10(p.brunt_N/(2*np.pi)), 
                                              fill_value=np.nan, bounds_error=0)(xr) 
                                              for p in self.profs])
    
        conv = np.array([sp.interpolate.interp1d(10**p.logR, p.brunt_N<0, 
                    fill_value=np.nan, bounds_error=0)(xr) 
                for p in self.profs])


        ppcnomin = np.array([np.min(10**p.logR[(p.pp+p.cno) > 1]) for p in self.profs])
        ppcnomax = np.array([np.max(10**p.logR[(p.pp+p.cno) > 1]) for p in self.profs])
    
        triamin = np.array([np.min(10**p.logR[(p.tri_alpha) > 1]) for p in self.profs])
        triamax = np.array([np.max(10**p.logR[(p.tri_alpha) > 1]) for p in self.profs])
    
    
        plt.figure(figsize=(7,6.5))

        if ax is None:
            ax = plt.gca()
    
        norm = mpl.colors.Normalize(vmin=-6, vmax=0)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
        vmin = int(norm.vmin)
        vmax = int(norm.vmax)
    
        ax.contourf(Y, X, conv, levels=[0,1,2,3,4,5,6,7], vmin=-1, vmax=3, cmap='Greys', zorder=-99999)
        ax.contourf(Y, X, Z, levels=np.arange(-6, 1, 1), vmin=vmin, vmax=vmax, cmap='Greens', zorder=-99999)

        ax.set_rasterization_zorder(-1)
    
        ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
        ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)
    
        specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF.log_Teff])
    

        # ax.plot([ages[specidx==cidx] for cidx in np.unique(specidx)], [10**self.DF.log_R[specidx==cidx] for cidx in np.unique(specidx)],
        #          c='w', lw=10, zorder=-999)
        # ax.plot([ages[specidx==cidx]  for cidx in np.unique(specidx)], [10**self.DF.log_R[specidx==cidx] for cidx in np.unique(specidx)],
        #          c=[rgbs[cidx] for cidx in np.unique(specidx)], lw=8, zorder=-99)

        for cidx in np.unique(specidx):
            idx = specidx == cidx
            for ii, x in enumerate(idx):
                if not x and ii+1<len(idx) and idx[ii+1]:
                    idx[ii] = 1
            ax.plot(ages[idx], 10**self.DF.log_R[idx], c='w', lw=10, zorder=-999)
            ax.plot(ages[idx], 10**self.DF.log_R[idx], c=rgbs[cidx], lw=8, zorder=-99)
    
        # radii = [[prof.logR[np.argmin((10**prof.logR - radius)**2)] 
        #         if radius<10**prof.logR[0] else np.nan 
        #         for prof in self.profs]
        #         for radius in [0.01, 0.05, 0.1, 0.5, 1, 10]]#, 100]]
        # for rad in radii:
        #     ax.plot(ages, rad, ls='-', c='k', lw=1, zorder=-99)

        # core_radius = [10**prof.logR[np.argmin()] for prof in self.profs]
    
        if mdl is not None:
            #plt.axvline(track.get_history(mdl).star_age.values[0]/1e6, ls='--', c='gray')
            ax.plot(self.get_history(mdl).star_age.values[0]/1e9, 
                    10**self.get_history(mdl).log_R.values[0]+1, marker='*', ms=20, c='k')
    
        if nolabels is False:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'Fractional radius $\mathbf{r/R_\odot}$')
        # ax.set_ylim(r_min, r_max * 1.1)
        ax.set_ylim(lower_y, r_max*1.1)
        #plt.gca().set_yticks([0, 5, 10, 15])
    
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if show_colorbar is True:
            cb = plt.colorbar(cmap, label=r'Buoyancy frequency $\mathbf{log~N}$/Hz',
                        boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        ticks=np.array(range(vmin, vmax+1, 1)),
                        ax=ax)
            cb.ax.minorticks_off()

        ax.set_title(title)
        # ax.set_yscale('log')
    
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.151)

        #plt.show()

    
    def plot_kippenhahn_composition_he(self, xlim=None, xlabel='Age [Gyr]', mdl=None, ax=None, title=None, show_colorbar=True, nolabels=False, lower_y=0.005):
        r_min = np.min(np.array([np.min(10**prof.logR) for prof in self.profs]))
        r_max = np.max(np.array([np.max(10**prof.logR) for prof in self.profs]))
        xr = np.linspace(r_min, r_max, 10000)

        ages = np.array([self.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in self.index.profile_number])
        
        X, Y = np.meshgrid(xr, ages)
      
        
        Z_he = np.array([sp.interpolate.interp1d(10**p.logR, p.y_mass_fraction_He, fill_value=np.nan, bounds_error=0)(xr) for p in self.profs])


        ppcnomin = np.array([np.min(10**p.logR[(p.pp+p.cno) > 1]) for p in self.profs])
        ppcnomax = np.array([np.max(10**p.logR[(p.pp+p.cno) > 1]) for p in self.profs])
    
        triamin = np.array([np.min(10**p.logR[(p.tri_alpha) > 1]) for p in self.profs])
        triamax = np.array([np.max(10**p.logR[(p.tri_alpha) > 1]) for p in self.profs])
    
    
        plt.figure(figsize=(7,6.5))

        if ax is None:
            ax = plt.gca()
    
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap_he = mpl.cm.ScalarMappable(cmap=mpl.cm.Blues)
        cmap_h = mpl.cm.ScalarMappable(cmap=mpl.cm.Oranges)
        vmin = int(norm.vmin)
        vmax = int(norm.vmax)
    
        ax.contourf(Y, X, Z_he, 
                    # levels=np.arange(0, 1, 5),
                      vmin=vmin, vmax=vmax, cmap='Blues', zorder=-99999)

        ax.set_rasterization_zorder(-1)
    
        ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
        ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)
    
        specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF.log_Teff])
        for cidx in np.unique(specidx):
            idx = specidx == cidx
            for ii, x in enumerate(idx):
                if not x and ii+1<len(idx) and idx[ii+1]:
                    idx[ii] = 1
            ax.plot(ages[idx], 10**self.DF.log_R[idx], c='w', lw=10, zorder=-999)
            ax.plot(ages[idx], 10**self.DF.log_R[idx], c=rgbs[cidx], lw=8, zorder=-99)
    
    
        if mdl is not None:
            #plt.axvline(track.get_history(mdl).star_age.values[0]/1e6, ls='--', c='gray')
            ax.plot(self.get_history(mdl).star_age.values[0]/1e9, 
                    10**self.get_history(mdl).log_R.values[0]+1, marker='*', ms=20, c='k')
    
        if nolabels is False:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'Fractional radius $\mathbf{r/R_\odot}$')
        # ax.set_ylim(r_min, r_max * 1.1)
        ax.set_ylim(lower_y, r_max*1.1)
        #plt.gca().set_yticks([0, 5, 10, 15])
    
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if show_colorbar is True:
            cb = plt.colorbar(cmap_he, label=r'Helium Abundance',
                        # boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        # ticks=np.array(range(vmin, vmax+1, 1)),
                        ax=ax)
            cb.ax.minorticks_off()

        ax.set_title(title)
        ax.set_yscale('log')

    
    def plot_kippenhahn_composition_h(self, xlim=None, xlabel='Age [Gyr]', mdl=None, ax=None, title=None, show_colorbar=True, nolabels=False, lower_y=0.005):
        r_min = np.min(np.array([np.min(10**prof.logR) for prof in self.profs]))
        r_max = np.max(np.array([np.max(10**prof.logR) for prof in self.profs]))
        xr = np.linspace(r_min, r_max, 10000)

        ages = np.array([self.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in self.index.profile_number])
        
        X, Y = np.meshgrid(xr, ages)
        Z_h = np.array([sp.interpolate.interp1d(10**p.logR, p.x_mass_fraction_H, fill_value=np.nan, bounds_error=0)(xr) for p in self.profs])


        ppcnomin = np.array([np.min(10**p.logR[(p.pp+p.cno) > 1]) for p in self.profs])
        ppcnomax = np.array([np.max(10**p.logR[(p.pp+p.cno) > 1]) for p in self.profs])
    
        triamin = np.array([np.min(10**p.logR[(p.tri_alpha) > 1]) for p in self.profs])
        triamax = np.array([np.max(10**p.logR[(p.tri_alpha) > 1]) for p in self.profs])
    
    
        plt.figure(figsize=(7,6.5))

        if ax is None:
            ax = plt.gca()
    
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap_he = mpl.cm.ScalarMappable(cmap=mpl.cm.Blues)
        cmap_h = mpl.cm.ScalarMappable(cmap=mpl.cm.Oranges)
        vmin = int(norm.vmin)
        vmax = int(norm.vmax)
    
        ax.contourf(Y, X, Z_h,
                    #  levels=np.arange(0, 1, 5), 
                     vmin=vmin, vmax=vmax, cmap='Oranges', zorder=-99999)

        ax.set_rasterization_zorder(-1)
    
        ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
        ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)
    
        specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF.log_Teff])
        for cidx in np.unique(specidx):
            idx = specidx == cidx
            for ii, x in enumerate(idx):
                if not x and ii+1<len(idx) and idx[ii+1]:
                    idx[ii] = 1
            ax.plot(ages[idx], 10**self.DF.log_R[idx], c='w', lw=10, zorder=-999)
            ax.plot(ages[idx], 10**self.DF.log_R[idx], c=rgbs[cidx], lw=8, zorder=-99)
    
    
        if mdl is not None:
            #plt.axvline(track.get_history(mdl).star_age.values[0]/1e6, ls='--', c='gray')
            ax.plot(self.get_history(mdl).star_age.values[0]/1e9, 
                    10**self.get_history(mdl).log_R.values[0]+1, marker='*', ms=20, c='k')
    
        if nolabels is False:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'Fractional radius $\mathbf{r/R_\odot}$')
        # ax.set_ylim(r_min, r_max * 1.1)
        ax.set_ylim(lower_y, r_max*1.1)
        #plt.gca().set_yticks([0, 5, 10, 15])
    
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if show_colorbar is True:
            cb = plt.colorbar(cmap_h, label=r'Hydrogen Abundance',
                        # boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        # ticks=np.array(range(vmin, vmax+1, 1)),
                        ax=ax)
            cb.ax.minorticks_off()

        ax.set_title(title)
        ax.set_yscale('log')
    

    
    def plot_kippenhahn_composition_h_mass(self, xlim=None, xlabel='Age [Gyr]', mdl=None, ax=None, title=None, show_colorbar=True, nolabels=False, lower_y=0.005):
        mass_max = np.min(np.array([np.max(prof.mass) for prof in self.profs]))
        print(mass_max)
        xm = np.linspace(0, mass_max, 10000)

        ages = np.array([self.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in self.index.profile_number])
        
        X, Y = np.meshgrid(xm, ages)
        Z_h = np.array([sp.interpolate.interp1d(p.mass, p.x_mass_fraction_H, fill_value=np.nan, bounds_error=0)(xm) for p in self.profs])


        ppcnomin = np.array([np.min(p.mass[(p.pp+p.cno) > 1]) for p in self.profs])
        ppcnomax = np.array([np.max(p.mass[(p.pp+p.cno) > 1]) for p in self.profs])
    
        triamin = np.array([np.min(p.mass[(p.tri_alpha) > 1]) for p in self.profs])
        triamax = np.array([np.max(p.mass[(p.tri_alpha) > 1]) for p in self.profs])
    
    
        plt.figure(figsize=(7,6.5))

        if ax is None:
            ax = plt.gca()
    
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap_he = mpl.cm.ScalarMappable(cmap=mpl.cm.Blues)
        cmap_h = mpl.cm.ScalarMappable(cmap=mpl.cm.Oranges)
        vmin = int(norm.vmin)
        vmax = int(norm.vmax)
    
        ax.contourf(Y, X, Z_h,
                    #  levels=np.arange(0, 1, 5), 
                     vmin=vmin, vmax=vmax, cmap='Oranges', zorder=-99999)

        ax.set_rasterization_zorder(-1)
    
        ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
        ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)
    
        specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF.log_Teff])
        for cidx in np.unique(specidx):
            idx = specidx == cidx
            for ii, x in enumerate(idx):
                if not x and ii+1<len(idx) and idx[ii+1]:
                    idx[ii] = 1
            ax.plot(ages[idx], self.DF.star_mass[idx], c='w', lw=10, zorder=-999)

            ax.plot(ages[idx], self.DF.star_mass[idx], c=rgbs[cidx], lw=8, zorder=-99)
    
    
        if mdl is not None:
            #plt.axvline(track.get_history(mdl).star_age.values[0]/1e6, ls='--', c='gray')
            ax.plot(self.get_history(mdl).star_age.values[0]/1e9, 
                    10**self.get_history(mdl).log_R.values[0]+1, marker='*', ms=20, c='k')
    
        if nolabels is False:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'Fractional mass $\mathbf{m/M_\odot}$')
        # ax.set_ylim(r_min, r_max * 1.1)
        # ax.set_ylim(lower_y, r_max*1.1)
        #plt.gca().set_yticks([0, 5, 10, 15])
    
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if show_colorbar is True:
            cb = plt.colorbar(cmap_h, label=r'Hydrogen Abundance',
                        # boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        # ticks=np.array(range(vmin, vmax+1, 1)),
                        ax=ax)
            cb.ax.minorticks_off()

        ax.set_title(title)
    

    def plot_kippenhahn_composition_he_mass(self, xlim=None, xlabel='Age [Gyr]', mdl=None, ax=None, title=None, show_colorbar=True, nolabels=False, lower_y=0.005):
        mass_max = np.min(np.array([np.max(prof.mass) for prof in self.profs]))
        # print(mass_max)
        xm = np.linspace(0, mass_max, 10000)

        ages = np.array([self.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in self.index.profile_number])
        
        X, Y = np.meshgrid(xm, ages)
        Z_he = np.array([sp.interpolate.interp1d(p.mass, p.y_mass_fraction_He, fill_value=np.nan, bounds_error=0)(xm) for p in self.profs])

        ppcnomin = np.array([np.min(p.mass[(p.pp+p.cno) > 1]) for p in self.profs])
        ppcnomax = np.array([np.max(p.mass[(p.pp+p.cno) > 1]) for p in self.profs])
    
        triamin = np.array([np.min(p.mass[(p.tri_alpha) > .1]) for p in self.profs])
        triamax = np.array([np.max(p.mass[(p.tri_alpha) > .1]) for p in self.profs])
    
    
    
        plt.figure(figsize=(7,6.5))

        if ax is None:
            ax = plt.gca()
    
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap_he = mpl.cm.ScalarMappable(cmap=mpl.cm.Blues)
        cmap_h = mpl.cm.ScalarMappable(cmap=mpl.cm.Oranges)
        vmin = int(norm.vmin)
        vmax = int(norm.vmax)
    
        ax.contourf(Y, X, Z_he,
                    #  levels=np.arange(0, 1, 5), 
                     vmin=vmin, vmax=vmax, cmap='Blues', zorder=-99999)

        ax.set_rasterization_zorder(-1)
    
        ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
        ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)
    
        specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF.log_Teff])
        for cidx in np.unique(specidx):
            idx = specidx == cidx
            for ii, x in enumerate(idx):
                if not x and ii+1<len(idx) and idx[ii+1]:
                    idx[ii] = 1
            ax.plot(ages[idx], self.DF.star_mass[idx], c='w', lw=10, zorder=-999)

            ax.plot(ages[idx], self.DF.star_mass[idx], c=rgbs[cidx], lw=8, zorder=-99)
    
    
        if mdl is not None:
            #plt.axvline(track.get_history(mdl).star_age.values[0]/1e6, ls='--', c='gray')
            ax.plot(self.get_history(mdl).star_age.values[0]/1e9, 
                    10**self.get_history(mdl).log_R.values[0]+1, marker='*', ms=20, c='k')
    
        if nolabels is False:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'Fractional mass $\mathbf{m/M_\odot}$')
        # ax.set_ylim(r_min, r_max * 1.1)
        # ax.set_ylim(lower_y, r_max*1.1)
        #plt.gca().set_yticks([0, 5, 10, 15])
    
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if show_colorbar is True:
            cb = plt.colorbar(cmap_he, label=r'Helium Abundance',
                        # boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        # ticks=np.array(range(vmin, vmax+1, 1)),
                        ax=ax)
            cb.ax.minorticks_off()

        ax.set_title(title)
        
    def plot_kippenhahn_composition_z_mass(self, xlim=None, xlabel='Age [Gyr]', mdl=None, ax=None, title=None, show_colorbar=True, nolabels=False, lower_y=0.005):
        mass_max = np.min(np.array([np.max(prof.mass) for prof in self.profs]))
        # print(mass_max)
        xm = np.linspace(0, mass_max, 10000)

        ages = np.array([self.get_history(prof_num).star_age.values[0]/1e9 
                     for prof_num in self.index.profile_number])
        
        X, Y = np.meshgrid(xm, ages)
        Z_z = np.array([sp.interpolate.interp1d(p.mass, p.z_mass_fraction_metals, fill_value=np.nan, bounds_error=0)(xm) for p in self.profs])


        ppcnomin = np.array([np.min(p.mass[(p.pp+p.cno) > 1]) for p in self.profs])
        ppcnomax = np.array([np.max(p.mass[(p.pp+p.cno) > 1]) for p in self.profs])
    
        triamin = np.array([np.min(p.mass[(p.tri_alpha) > .1]) for p in self.profs])
        triamax = np.array([np.max(p.mass[(p.tri_alpha) > .1]) for p in self.profs])
    
    
    
        plt.figure(figsize=(7,6.5))

        if ax is None:
            ax = plt.gca()
    
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(cmap=mpl.cm.Purples)
        vmin = int(norm.vmin)
        vmax = int(norm.vmax)
    
        ax.contourf(Y, X, Z_z,
                    #  levels=np.arange(0, 1, 5), 
                     vmin=vmin, vmax=vmax, cmap='Purples', zorder=-99999)

        ax.set_rasterization_zorder(-1)
    
        ax.fill_between(ages, ppcnomin, ppcnomax, hatch='\\\\', ec='k', fc='none', alpha=0.8, lw=0, zorder=-9999)
        ax.fill_between(ages, triamin,  triamax,  hatch='////', ec='k', fc='none', alpha=1,   lw=0, zorder=-9999)
    
        specidx = np.array([np.min(np.where(Teff > spectrals)) for Teff in 10**self.DF.log_Teff])
        for cidx in np.unique(specidx):
            idx = specidx == cidx
            for ii, x in enumerate(idx):
                if not x and ii+1<len(idx) and idx[ii+1]:
                    idx[ii] = 1
            ax.plot(ages[idx], self.DF.star_mass[idx], c='w', lw=10, zorder=-999)

            ax.plot(ages[idx], self.DF.star_mass[idx], c=rgbs[cidx], lw=8, zorder=-99)
    
    
        if mdl is not None:
            #plt.axvline(track.get_history(mdl).star_age.values[0]/1e6, ls='--', c='gray')
            ax.plot(self.get_history(mdl).star_age.values[0]/1e9, 
                    10**self.get_history(mdl).log_R.values[0]+1, marker='*', ms=20, c='k')
    
        if nolabels is False:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'Fractional mass $\mathbf{m/M_\odot}$')
        # ax.set_ylim(r_min, r_max * 1.1)
        # ax.set_ylim(lower_y, r_max*1.1)
        #plt.gca().set_yticks([0, 5, 10, 15])
    
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if show_colorbar is True:
            cb = plt.colorbar(cmap, label=r'Z Abundance',
                        # boundaries=np.array(range(vmin, vmax+2, 1))-0.5,
                        # ticks=np.array(range(vmin, vmax+1, 1)),
                        ax=ax)
            cb.ax.minorticks_off()

        ax.set_title(title)
    


    def plot_echelle(self, profile_number, sph_deg=-1, rad_ord=-1):
        ell_label = {0: 'radial', 1: 'dipole', 2: 'quadrupole', 3: 'octupole'}

        hist = self.get_history(profile_number)
        prof = self.profs[profile_number-1]
        freq = self.freqs[profile_number-1]

        freq = freq[freq.n_g == 0]
    
        nu_max   = hist.nu_max.values[0]
        radial = freq[freq.l == 0]
        Dnu = np.median(np.diff(radial['Re(freq)'].values))
    
        colors = ('black', 'red', 'blue', 'purple')
        for ell in np.unique(freq.l.values):
            nus = freq[freq.l == ell]
            plt.plot(nus['Re(freq)'] % Dnu,
                    nus['Re(freq)'], '.', 
                    mfc=colors[ell], mec='white', alpha=0.85,
                    ms=15, mew=1, 
                    label=str(ell) + ' (' + ell_label[ell] + ')')#r'$\ell=' + str(ell) + r'$')
    
        if sph_deg >= 0 and rad_ord >= 0:
            freq = freq[np.logical_and(freq.l == sph_deg, freq.n_pg == rad_ord)]
            if len(freq) > 0:
                plt.plot(freq['Re(freq)'] % Dnu, freq['Re(freq)'], 'o', zorder=-99, mec='k', ms=10, mfc='w')
    
        plt.legend(loc='lower right', title=r'spherical degree $\ell$', fontsize=12)
    
        plt.axvline(Dnu, ls='--', c='darkgray', zorder=-99)
        plt.axhline(nu_max, ls='--', c='darkgray', zorder=-99)
    
    
        plt.ylabel(r'frequency $\nu/\mu\rm{Hz}$')
        plt.xlabel(r'$\nu\; \rm{mod}\; \Delta\nu/\mu\rm{Hz}$')
        plt.title('Echelle Diagram', size=24)


    def plot_HR_pg(self, cbar=True, c=None, alpha=0.5, label=None, offset=0, unstable_only=False):
        
        ts = np.array([10**self.get_value_profile('log_Teff', prof) for prof in self.index.profile_number])
        Ls = np.array([10**self.get_value_profile('log_L', prof) for prof in self.index.profile_number])
        pgs = np.array([self.get_Pg(prof) for prof in self.index.profile_number])

        if unstable_only is True:
            ts = ts[self.unstable_profs]
            Ls = Ls[self.unstable_profs]
            pgs = pgs[self.unstable_profs]

        plt.plot(ts, Ls + offset, c=c, label=label, alpha=alpha)

        plt.scatter(ts, Ls + offset, c=pgs)

        plt.gca().invert_xaxis()
        plt.xlabel(r'Effective Temperature $T_{\rm{eff}}$ (K)')
        plt.ylabel(r'Luminosity $L/\rm{L}_\odot$')
        plt.yscale('log')
        
        if cbar:
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Period Spacing')
 

    def plot_pg(self, title=None, c=None, label=None, ax=None, crop=False):
        if ax is None:
            ax = plt.gca()

        # pgs = np.array([self.get_Pg(prof) for prof in self.index.profile_number])
        # # ts = np.array([10**self.get_history(prof).log_Teff for prof in self.index.profile_number])
        # ts = np.array([10**self.DF.log_Teff])

        # ax.scatter(ts, pgs, color=c, label=label)

        if crop is False:
            ax.scatter([10**self.DF.log_Teff], [self.get_Pg(prof) for prof in self.index.profile_number], color=c, label=label, s=5)
        if crop is not False:
            df = self.DF[0:crop]
            profs = self.index.profile_number[0:crop]
            ax.scatter([10**df.log_Teff], [self.get_Pg(prof) for prof in profs], color=c, label=label, s=5)


        ax.set_xlabel('Teff (K)')
        ax.set_ylabel('Period Spacing (s)')
        ax.invert_xaxis()

    def get_fundamental_period(self, profile_num):
        # PERIOD IN HOURS
        
        freqs = self.freqs[profile_num-1]

        # get fundamental: l=0, n_p=1
        radial = freqs[freqs['l']==0]
        if radial.empty: 
            radial = freqs[freqs['l']=='0']

        f = radial[radial['n_p']==1]
        if f.empty:
            f = radial[radial['n_p']=='1']

        p = [1/(float(f['Re(freq)']) * 10**-6) / 3600][0]

        return p


    def plot_fundamental(self, max=None, title=None, c=None, label=None, ax=None, crop=False):
        if ax is None:
            ax = plt.gca()
        if max is None:
            max = len(self.freqs)-1
        
        fundamentals = []
        ages = []
        for i in range(1, max):
            try:
                fundamentals.append(self.get_fundamental_period(i))
                ages.append(10**-9*self.DF['star_age'][i])
            except:
                pass
 


        # ax.scatter([self.get_fundamental_period(prof)[0] for prof in np.arange(2, max)], [self.get_Pg(prof) for prof in np.arange(2, max)],
        #  color=c, label=label, s=5)
        ax.scatter(ages, fundamentals, color=c, label=label, s=5)
        # plt.scatter([10**-9*nv_premix.DF['star_age'][p] for p in profs], [nv_premix.get_fundamental_period(p) for p in profs], color='b', s=2, label='premix')



        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel('Fundamental Period (hours)')


    def plot_pg_vs_fundamental(self, max=None, title=None, c=None, label=None, ax=None, crop=False, spline=False, lam=None, nad=False):
        if ax is None:
            ax = plt.gca()
        if max is None:
            max = len(self.freqs)-1
        
        fundamentals = []
        pgs = []
        for i in range(1, max):
            try:
                if np.abs(self.get_Pg(i) - self.get_Pg(i-1)) <= 3: 
                    fundamentals.append(self.get_fundamental_period(i))
                    pgs.append(self.get_Pg(i))
            except:
                pass
 

        # ax.scatter([self.get_fundamental_period(prof)[0] for prof in np.arange(2, max)], [self.get_Pg(prof) for prof in np.arange(2, max)],
        #  color=c, label=label, s=5)

        start = next(x for x, val in enumerate(fundamentals) if val > 10)


        if spline is True:
            x = np.linspace(10, fundamentals[-1])
            spl = make_smoothing_spline(fundamentals[start:], pgs[start:], lam=lam)

            plt.plot(x, spl(x), label=label, color=c, linewidth=3)
        if spline is False:
            if nad is True:
                ax.scatter(fundamentals, pgs, color=c, alpha=0.1, s=5)
            else:
                ax.scatter(fundamentals, pgs, color=c, label=label, s=5)
        if nad is True:
            fundamentals_nad = []
            pgs_nad = []
            for i in range(1, max):
                if i in self.unstable_profs:
                    try:
                        if np.abs(self.get_Pg(i) - self.get_Pg(i-1)) <= 3: 
                            fundamentals_nad.append(self.get_fundamental_period(i))
                            pgs_nad.append(self.get_Pg(i))
                    except:
                        pass
                            
            ax.scatter(fundamentals_nad, pgs_nad, color=c, label=label, s=5)


        ax.set_xlabel('Fundamental Period (hours)')
        ax.set_ylabel('Period Spacing (s)')



        
    def plot_pg_rate_vs_fundamental(self, max=None, title=None, c=None, label=None, ax=None, crop=False, spline=False, lam=None, nad=False):
        if ax is None:
            ax = plt.gca()
        if max is None:
            max = len(self.freqs)-2
        
        fundamentals = np.zeros(max)
        pgs = np.zeros(max)
        for i in range(1, max):
            try:
                if np.abs(self.get_Pg(i) - self.get_Pg(i-1)) <= 3: 
                    fundamentals[i] = (self.get_fundamental_period(i))
                    pgs[i] = (self.get_fundamentalperiodchange_rate(i))
            except:
                fundamentals[i] = np.nan
                pgs[i] = np.nan
 

        # ax.scatter([self.get_fundamental_period(prof)[0] for prof in np.arange(2, max)], [self.get_Pg(prof) for prof in np.arange(2, max)],
        #  color=c, label=label, s=5)

        start = next(x for x, val in enumerate(fundamentals) if val > 10)


        if spline is True:
            x = np.linspace(10, fundamentals[-1])
            spl = make_smoothing_spline(fundamentals[start:], pgs[start:], lam=lam)

            plt.plot(x, spl(x), label=label, color=c, linewidth=3)

        if spline is False:
            if nad is True:
                ax.scatter(fundamentals, pgs, color=c, alpha=0.1, s=5)
            else:
                ax.scatter(fundamentals, pgs, color=c, label=label, s=5)
        
        if nad is True:
            fundamentals_nad = np.zeros(max)
            pgs_nad = np.zeros(max)
            for i in range(1, max):
                if i in self.unstable_profs:
                    try:
                        if np.abs(self.get_Pg(i) - self.get_Pg(i-1)) <= 3: 
                            fundamentals_nad[i] = (self.get_fundamental_period(i))
                            pgs_nad[i] = (self.get_fundamentalperiodchange_rate(i))
                    except:
                        fundamentals_nad[i] = np.nan
                        pgs_nad[i] = np.nan
                            
            ax.scatter(fundamentals_nad, pgs_nad, color=c, label=label, s=5)


        ax.set_xlabel('Fundamental Period (hours)')
        ax.set_ylabel('Fundamental Period Change Rate (s/yr)')


        

    def plot_pg_vs_yc(self, max=None, title=None, c=None, label=None, ax=None, crop=False, spline=False, lam=None, nad=False):
        if ax is None:
            ax = plt.gca()
        if max is None:
            max = len(self.freqs)-1
        
        ycs = []
        pgs = []
        for i in range(1, max):
            try:
                if np.abs(self.get_Pg(i) - self.get_Pg(i-1)) <= 3: 
                    ycs.append(self.profs[i].y_mass_fraction_He.iloc[-1])
                    pgs.append(self.get_Pg(i))
            except:
                pass
 

        # ax.scatter([self.get_fundamental_period(prof)[0] for prof in np.arange(2, max)], [self.get_Pg(prof) for prof in np.arange(2, max)],
        #  color=c, label=label, s=5)

        # start = next(x for x, val in enumerate(ycs) if val > 10)



        if spline is False:
            if nad is True:
                ax.scatter(ycs, pgs, color=c, alpha=0.1, s=5)
            else:
                ax.scatter(ycs, pgs, color=c, label=label, s=5)
        if nad is True:
            ycs_nad = []
            pgs_nad = []
            for i in range(1, max):
                if i in self.unstable_profs:
                    try:
                        if np.abs(self.get_Pg(i) - self.get_Pg(i-1)) <= 3: 
                            ycs_nad.append(self.profs[i].y_mass_fraction_He.iloc[-1])
                            pgs_nad.append(self.get_Pg(i))
                    except:
                        pass
                            
            ax.scatter(ycs_nad, pgs_nad, color=c, label=label, s=5)


        ax.set_xlabel('Core Helium Mass Fraction')
        ax.set_ylabel('Period Spacing (s)')
        ax.set_xlim([0, 0.3])

    
    def plot_fundamental_vs_teff(self, max=None, title=None, c=None, label=None, ax=None, crop=False):
        if ax is None:
            ax = plt.gca()
        if max is None:
            max = len(self.freqs)-1
        
        fundamentals = []
        teffs = []
        for i in range(1, max):
            try:
                fundamentals.append(self.get_fundamental_period(i))
                teffs.append(10**self.DF['log_Teff'][i])
            except:
                pass
 
        ax.scatter(teffs, fundamentals, color=c, label=label, s=5)


        ax.set_xlabel('Teff (K)')
        ax.set_ylabel('Fundamental Period (hours)')



    def plot_pg_vs_dnu(self, title=None, c=None, label=None, ax=None):
        if ax is None:
            ax = plt.gca()

        # pgs = np.array([self.get_Pg(prof) for prof in self.index.profile_number])
        # # ts = np.array([10**self.get_history(prof).log_Teff for prof in self.index.profile_number])
        # ts = np.array([10**self.DF.log_Teff])

        # ax.scatter(ts, pgs, color=c, label=label)

        ax.scatter([self.DF.delta_nu], [self.get_Pg(prof) for prof in tqdm(self.index.profile_number)], color=c, label=label, s=5)
        ax.set_xlabel(r'$\Delta\nu$')
        ax.set_ylabel('Period Spacing (s)')

    
    def plot_period_spacing_vs_n(self, profile_num, ax=None, xlim=None, ylim=None,title=None, c='blue', label=None, plot_pg=True):
        if ax is None:
            ax = plt.gca()

        x = []
        y = []

        df = self.freqs[profile_num]

        for i, f in enumerate(df['Re(freq)']):
            if i == 0:
                continue
            period_i = 1/(f*10**-6)  # from uHz to Hz
            period_previous = 1/(df['Re(freq)'][i-1] *10**-6 )

            x.append(df['n_pg'][i-1])
            y.append(-period_i + period_previous)

        plt.scatter(x, y, color=c, label=label)
        plt.plot(x, y, color=c, label=label)

        if plot_pg is True:
            plt.axhline(y=self.get_Pg(profile_num), color='red', linestyle='dashed', zorder=-999)
        plt.xlabel('n_pg')
        plt.ylabel('Period Spacing (s)')
        plt.xlim(xlim)
        plt.ylim(ylim)

    def plot_period_spacing_vs_npg(self, profile_num, ax=None, plot_pg=True, c='blue', age=True, xlim=None, ylim=None):
        if ax is None:
            ax = plt.gca()

        def get_DP(freqk):
            freqk = freqk[freqk.l == 1]
            freqk = freqk.sort_values(by=['n_p', 'l', 'n_g'], ascending=[True, True, False])
            freqk['P'] = (1 / (freqk['Re(freq)'] * 10**-6)) / 3600  # convert to hours
            freqk['DP'] = freqk['P'].diff().abs() * 60 * 60  # convert to minutes
            freqk['adj'] = freqk['n_g'].diff(1).abs() == 1
            freqk['DP'] = freqk.apply(lambda x: x['DP'] if x['adj'] else None, axis=1)
            return freqk
         
        gS = get_DP(self.freqs[profile_num-1])
        ax.plot(gS.P *3600 % self.get_Pg(profile_num), gS.P, '.-', color=c)
         
        if plot_pg is True:
            ax.axhline(y=self.get_Pg(profile_num), color='red', linestyle='dashed', zorder=-999)
        if age is True:
            ax.text(x=0.75, y=0.9, s=f'Age: {10**-9*self.get_value_profile('star_age', profile_num):.2f} Gyr',
                     transform=plt.gca().transAxes)
        ax.set_xlabel('Period (hours)')
        ax.set_ylabel('Period Spacing (s)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    
    def plot_period_spacing_vs_f(self, profile_num, ax=None, title=None, c='blue', label=None, plot_pg=True):
        if ax is None:
            ax = plt.gca()

        x = []
        y = []

        df = self.freqs[profile_num]

        for i, f in enumerate(df['Re(freq)']):
            if i == 0:
                continue
            period_i = (1/(f*10**-6))  # from uHz to Hz to s to min
            period_previous = (1/(df['Re(freq)'][i-1] *10**-6 ))

            x.append(df['Re(freq)'][i])
            y.append(-period_i + period_previous)

        plt.scatter(x, y, color=c, label=label)
        plt.plot(x, y, color=c, label=label)

        if plot_pg is True:
            plt.axhline(y=self.get_Pg(profile_num), color='red', linestyle='dashed', zorder=-999)
        plt.xlabel('Frequency (uHz)')
        plt.ylabel('Period Spacing (s)')

    
    def plot_period_spacing_vs_p(self, profile_num, ax=None, plot_pg=True, c='blue', age=True, xlim=None, ylim=None):
        if ax is None:
            ax = plt.gca()

        def get_DP(freqk):
            freqk = freqk[freqk.l == 1]
            freqk = freqk.sort_values(by=['n_p', 'l', 'n_g'], ascending=[True, True, False])
            freqk['P'] = (1 / (freqk['Re(freq)'] * 10**-6)) / 3600  # convert to hours
            freqk['DP'] = freqk['P'].diff().abs() * 60 * 60  # convert to minutes
            freqk['adj'] = freqk['n_g'].diff(1).abs() == 1
            freqk['DP'] = freqk.apply(lambda x: x['DP'] if x['adj'] else None, axis=1)
            return freqk
         
        gS = get_DP(self.freqs[profile_num-1])
        ax.plot(gS.P, gS.DP, '.-', color=c)
         
        if plot_pg is True:
            ax.axhline(y=self.get_Pg(profile_num), color='red', linestyle='dashed', zorder=-999)
        if age is True:
            ax.text(x=0.75, y=0.9, s=f'Age: {10**-9*self.get_value_profile('star_age', profile_num):.2f} Gyr',
                     transform=plt.gca().transAxes)
        ax.set_xlabel('Period (hours)')
        ax.set_ylabel('Period Spacing (s)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    
    def plot_petersen_diagram(self, profile_num, l, n=None, ax=None, xlim=None, ylim=None, c='blue', alpha=1):
        if ax is None:
            ax = plt.gca()
        
        freqs = self.freqs[profile_num-1]

        def p(f):
            return float(1/(f['Re(freq)'] * 10**-6) / 60)  # minutes

        # get first overtone: l=0, n=2
        radial = freqs[freqs['l']==0]
        try:
            f_10 = radial[radial['n_p']==2]
            p_10 = p(f_10)
        except:
            print(f'l = 0 , n = 2 modes do not exist in profile {profile_num}')
            pass


        f_l = freqs[freqs['l']==l]

        if n is not None:
            try:
                p_n = p(f_l[f_l['n_p']==n])
                ax.scatter(p_10, p_10/p_n, color=c, alpha=alpha)
            except:
                pass
                # print(f'l = {l}, n = {n} modes do not exist in profile {profile_num}')

        if n is None:
            colors = mpl.cm.viridis(np.linspace(0, 1, len(f_l)-1))
            for i in range(len(f_l)-1):
                # print(f_l.iloc[i])
                # print(p(f_l[i]))
                ax.scatter(p_10, p_10/p(f_l.iloc[i]), color=colors[i], alpha=alpha)
            
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        # plt.scatter(p_10, p_10/p_n, color=c)

        # p_other = 1/(freqs * 10**-6) / 60
        # plt.scatter(p_10, p_10/p_other, color=c)
        ax.set_xlabel('l=0, n=2 (min)')
        ax.set_ylabel('px/p(l=0, n=2)')


    def plot_petersen_diagram_g(self, profile_num, l, n=None, n_p=2, ax=None, xlim=None, ylim=None, c='blue', alpha=1):
        if ax is None:
            ax = plt.gca()
        
        freqs = self.freqs[profile_num-1]

        def p(f):
            return float(1/(f['Re(freq)'] * 10**-6) / 60)  # minutes

        # get first overtone: l=0, n=2
        radial = freqs[freqs['l']==0]
        try:
            f_10 = radial[radial['n_p']==n_p]
            p_10 = p(f_10)
        except:
            pass 
            # print(f'l = 0 , n = {n_p} modes do not exist in profile {profile_num}')


        f_l = freqs[freqs['l']==l]

        if n is not None:
            try:         
                f_l_p0 = f_l[f_l['n_p']==0]

                try:
                    p_n = p(f_l_p0[f_l_p0['n_g']==n])
                    ax.scatter(p_10, p_n/p_10, color=c, alpha=alpha)
                except:
                    pass
                    # print(f'l = {l}, n_g = {n}, n_p=0 modes do not exist in profile {profile_num}')
            except:
                print(f'there are no l={l} n_p=0 modes')

        if n is None:
            f_l_p0 = f_l[f_l['n_p']==0]

            colors = mpl.cm.viridis(np.linspace(0, 1, len(f_l_p0)-1))

            for i in range(len(f_l_p0)-1):
                # print(f_l.iloc[i])
                # print(p(f_l[i]))
                ax.scatter(p_10, p_10/p(f_l_p0.iloc[i]), color=colors[i], alpha=alpha)
            
        
        # ax.set_xlabel('l=0, n=2 (min)')
        # ax.set_ylabel('px/p(l=0, n=2)')

    
    def plot_period_spacing_NEWANDIMPROVED(self, profile, xlim=None, ylim=None):
        freqs = self.freqs[profile-1]

        freqs.n_g = [int(x) for x in freqs.n_g]
        # try:
        dipole_g = freqs[np.logical_and(freqs.l == 1, freqs.n_g > 0)] 
        # except:
            # sometimes the numbers are strings L
            # dipole_g = freqs[np.logical_and(freqs.l == 1, int(freqs.n_g)]

        dipole_g['P']  = 1/(dipole_g['Re(freq)'] * 10**-6) # seconds


        for mode in dipole_g.iterrows():
            # n_g  = mode[1]['n_g']
            # n_g2 = dipole_g[dipole_g['n_g'] == n_g+1]
            # if not n_g2.empty:
            #     dP = (n_g2['P'].values[0] - mode[1]['P']) # period spacing in seconds 
            #     dipole_g.loc[mode[0], 'dP'] = dP
            n_pg = mode[1]['n_pg']
            n_pg2 = dipole_g[dipole_g['n_pg'] == n_pg-1]

            if not n_pg2.empty:
                dP = (n_pg2['P'].values[0] - mode[1]['P']) # seconds
                dipole_g.loc[mode[0], 'dP'] = dP
        
        plt.plot(dipole_g['P']/3600, dipole_g['dP'], 
             'b.--', ms=10, zorder=10, ls=':', mec='k', mfc='k', label='exact')
        plt.axhline(self.get_Pg(profile), ls='--', c='k', label='asymptotic')
        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.xlabel('Period (hours)')
        plt.ylabel('Period Spacing (s)')

    def frequency_vs_periodmodperiodspacing(self, profile, xlim=None, ylim=None):
        freqs = self.freqs[profile-1]

        freqs.n_g = [int(x) for x in freqs.n_g]
        # try:
        dipole_g = freqs[np.logical_and(freqs.l == 1, freqs.n_g > 0)] 
        # except:
            # sometimes the numbers are strings L
            # dipole_g = freqs[np.logical_and(freqs.l == 1, int(freqs.n_g)]

        dipole_g['P']  = 1/(dipole_g['Re(freq)'] * 10**-6) # seconds


        for mode in dipole_g.iterrows():
            # n_g  = mode[1]['n_g']
            # n_g2 = dipole_g[dipole_g['n_g'] == n_g+1]
            # if not n_g2.empty:
            #     dP = (n_g2['P'].values[0] - mode[1]['P']) # period spacing in seconds 
            #     dipole_g.loc[mode[0], 'dP'] = dP
            n_pg = mode[1]['n_pg']
            n_pg2 = dipole_g[dipole_g['n_pg'] == n_pg-1]

            if not n_pg2.empty:
                dP = (n_pg2['P'].values[0] - mode[1]['P']) # seconds
                dipole_g.loc[mode[0], 'dP'] = dP
        
        # plt.plot(dipole_g['P']/3600, dipole_g['dP'], 
        #      'b.--', ms=10, zorder=10, ls=':', mec='k', mfc='k', label='exact')
        # plt.axhline(self.get_Pg(profile), ls='--', c='k', label='asymptotic')
        # plt.plot(dipole_g['P']/3600 % dipole_g['dP'], dipole_g['Re(freq)'])
        plt.plot(dipole_g['P']/3600 % self.get_Pg(profile), dipole_g['Re(freq)'])

        plt.xlim(xlim)
        plt.ylim(ylim)

        # plt.xlabel('Period (hours)')
        # plt.ylabel('Period Spacing (s)')

    def get_fundamentalperiodchange_rate(self, profile):
        # seconds/year
        return 3600*(self.get_fundamental_period(profile+1) - self.get_fundamental_period(profile)) / (self.get_value_profile('star_age', profile+1)- self.get_value_profile('star_age', profile))
    
        
    def propagation_dashedlines(self, profile):
        gyre = self.gyres[profile-1]
        N  = gyre.N
        x  = gyre.x
        N2 = gyre.N2

        # prof = self.profs[profile-1]

        # x_ = 10**prof.logR / np.max(10**prof.logR)
        # brunt = prof.brunt_N.values/(2*np.pi)*1e6
        # lamb  = prof.lamb_S.values*1e6/np.sqrt(2)*np.sqrt(1*(1+1))
        # plt.plot(x_, brunt, lw=3, label='Buoyancy')
        # delta_Pg = 2*np.pi**2/np.sqrt(2)/ sp.integrate.trapezoid(N[N>0]/x[N>0], x[N>0])


        Pi_0 = 2*np.pi**2/sp.integrate.trapezoid(gyre.N[gyre.N2>0]/gyre.x[gyre.N2>0], gyre.x[gyre.N2>0])
        Pi_r = 2*np.pi**2/sp.integrate.cumulative_trapezoid(N[gyre.N2>0]/gyre.x[gyre.N2>0], gyre.x[gyre.N2>0])
        Pi_r = np.concatenate(([Pi_r[0]], + Pi_r))

        buoy_rad = N[N2>0]/(10**-6 * 2*np.pi)

        plt.plot(Pi_0 / Pi_r, buoy_rad, 
                zorder=10, lw=3, label=r'Brunt–Väisälä',
                c='tab:blue')
        plt.xlabel(r'relative buoyancy radius $\Pi_0/\Pi_r$')
        #plt.ylabel(r'frequency $\nu/\mu\rm{Hz}$')
        plt.ylabel(r'period $P/\rm{hours}$')
        plt.xlim([0,1])
        ylim = [10, 10**5]
        plt.ylim(ylim)
        plt.semilogy()

        ax = plt.gca()

        yticks = [10**1, 10**2, 10**3, 10**4, 10**5]
        nutoP = lambda x: 1/(x * 10**-6) / (3600)
        ax.set_yticks(yticks)
        ax.set_yticklabels(['%2.2g' % nutoP(x) for x in yticks])

        plt.text(0.5, buoy_rad[np.where(Pi_0 / Pi_r >= 0.5)[0][0]]*1.2, r'Brunt–Väisälä $N$', ha='center',
                c='tab:blue', family='Latin Modern Sans')
        #plt.legend()
        
        freqs = self.freqs[profile-1]

        dipole_g = freqs[np.logical_and(freqs.l == 1, freqs.n_g > 0)] 

        dipole_g['P']  = 1/(dipole_g['Re(freq)'] * 10**-6) # seconds
        for mode in dipole_g.iterrows():
            n_g  = mode[1]['n_g']
            n_g2 = dipole_g[dipole_g['n_g'] == n_g+1]
            if not n_g2.empty:
                dP = (n_g2['P'].values[0] - mode[1]['P']) # period spacing in seconds 
                dipole_g.loc[mode[0], 'dP'] = dP

        for mode in dipole_g.iterrows():
            plt.axhline(mode[1]['Re(freq)'], ls='--', c='k', alpha=0.35)
            if mode[1]['n_g'] == 1:
                plt.text(0.5, mode[1]['Re(freq)']*1.2, r'dipolar modes ($\ell = 1$)', ha='center',
                    c='k', alpha=0.35, family='Latin Modern Sans')
            #plt.plot([0,1], [mode[1]['Re(freq)']]*2, ls='--', c='k')

        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        top.set_xlabel(r'fractional radius $r/R$', labelpad=10)

        find_pi = lambda x: gyre.x[N2>0][np.where(Pi_0/Pi_r > x)[0][0]]
        #xs = [0, 0.25, 0.5, 0.75, 0.99999]
        xs = [0, 0.2, 0.4, 0.6, 0.8, 0.99999]
        top.set_xticks(xs)
        top.set_xticklabels([r'$%0.2g$' % find_pi(x) if x<0.999 else r'$1.0$' for x in xs])

        ## plot overtones on the right y-axis 
        right = ax.twinx()
        right.set_ylim(ax.get_ylim())
        right.semilogy()
        right.set_ylabel(r'overtone $n_g$')
        right.yaxis.set_label_coords(1.1,0.25)

        per_spac = dipole_g[~np.isnan(dipole_g.dP)]
        # per_spac = per_spac[per_spac.P < 5]
        right.set_yticks(per_spac['Re(freq)'].values[::-1][[0,1,3,7, -1]])
        right.set_yticklabels([r'$%s$' % x for x in per_spac.n_g.values[::-1][[0,1,3,7, -1]]])

        # #right.set_yticks(dipole_g['Re(freq)'], minor=True)
        # #right.set_yticklabels([], minor=True)
        right.set_yticks([], minor=True)
        #right.set_yticklabels([], minor=True)



    def plot_convection_circles(self, profile_num, base_color=None, ax=None, age_plot=True, age_scale=10**-9, age_label='Gyr'):
        plt.clf()

        colors = {'O':  (175/255, 201/255, 1),
                'B': (199/255, 216/255, 1),
                'A': (1,244/255, 243/255),
                'F': (1, 229/255, 207/255),
                'G': (1, 217/255, 178/255),
                'K': (1, 199/255, 142/255),
                'M': (1, 166/255, 81/255)
                }
        
        if base_color is None:
            base_color = colors['M']

        plt.rcParams.update({ 'figure.figsize': (8,8)})

        if ax is None:
            ax=plt.gca()

        initial_mass = self.DF.iloc[0]['star_mass']

        df = self.DF.iloc[profile_num-1]
        mass = df['star_mass']
        age = df['star_age']

        teff = 10**df.log_Teff
        index = np.argmin(spectrals > teff) 
        surface_color = rgbs[index]

        base = plt.Circle((0,0), mass, color=base_color)
        conv_core = plt.Circle((0,0), df['mass_conv_core'], color='firebrick')
        conv1_top = plt.Circle((0,0), df['conv_mx1_top'] * mass, color='firebrick')
        conv1_bot = plt.Circle((0,0), df['conv_mx1_bot'] * mass, color=base_color)
        conv2_top = plt.Circle((0,0), df['conv_mx2_top'] * mass, color='firebrick')
        conv2_bot = plt.Circle((0,0), df['conv_mx2_bot'] * mass, color=base_color)

        p = self.profs[profile_num-1]
        ppcnomin = np.min(p.mass[(p.pp+p.cno) > 0.01]) 
        ppcnomax = np.max(p.mass[(p.pp+p.cno) > 0.01])
        triamin = np.min(p.mass[(p.tri_alpha) > 0.01])
        triamax = np.max(p.mass[(p.tri_alpha) > 0.01])
            
        h = mpl.patches.Annulus((0,0), ppcnomax, ppcnomax-ppcnomin, hatch='\\\\', alpha=0)
        he = mpl.patches.Annulus((0,0), triamax, triamax-triamin, hatch='////', alpha=0)

        surface = mpl.patches.Annulus((0,0), mass + 0.1, 0.05, color=surface_color)



        plt.gca().add_patch(base)
        plt.gca().add_patch(conv2_top)
        plt.gca().add_patch(conv2_bot)
            
        plt.gca().add_patch(conv1_top)
        plt.gca().add_patch(conv1_bot)

        plt.gca().add_patch(conv_core)

        plt.gca().add_patch(h)
        plt.gca().add_patch(he)

        plt.gca().add_patch(surface)

        if age_plot is True:
            plt.text(x=0.8, y=0.95, s=f'Age: {age_scale*self.get_value_profile('star_age', profile_num):.2f} {age_label}',
                     transform=plt.gca().transAxes)

                
        plt.xlim(-1.2 * initial_mass, 1.2 * initial_mass)
        plt.ylim(-1.2 * initial_mass, 1.2 * initial_mass)
        plt.xticks([])
        plt.yticks([])
        # plt.legend()



    def plot_convection_circles_radius(self, profile_num, base_color=None, ax=None, age_plot=True, xlim=None, age_scale=10**-9, age_label='Gyr'):
        plt.clf()

        colors = {'O':  (175/255, 201/255, 1),
                'B': (199/255, 216/255, 1),
                'A': (1,244/255, 243/255),
                'F': (1, 229/255, 207/255),
                'G': (1, 217/255, 178/255),
                'K': (1, 199/255, 142/255),
                'M': (1, 166/255, 81/255)
                }
        if base_color is None:
            base_color = colors['M']
            

        plt.rcParams.update({ 'figure.figsize': (8,8)})
        if ax is None:
            ax=plt.gca()

        df = self.DF.iloc[profile_num-1]
        radius = 10**df['log_R']
        # print(radius)

        max_r = max(10**self.DF['log_R'])

        
        teff = 10**df.log_Teff
        index = np.argmin(spectrals > teff) 
        surface_color = rgbs[index]

        base = plt.Circle((0,0), radius, color=base_color)

        # conv_core = plt.Circle((0,0), df['mass_conv_core'] * mass, color='firebrick')
        # conv1_top = plt.Circle((0,0), df['conv_mx1_top'] * mass, color='firebrick')
        # conv1_bot = plt.Circle((0,0), df['conv_mx1_bot'] * mass, color=base_color)
        # conv2_top = plt.Circle((0,0), df['conv_mx2_top'] * mass, color='firebrick')
        # conv2_bot = plt.Circle((0,0), df['conv_mx2_bot'] * mass, color=base_color)

        p = self.profs[profile_num-1]

        convection = 10**p.logR[p.brunt_N <= 0]
        # conv = mpl.patches.Annulus((0,0), max(convection), max(convection)-min(convection), color='firebrick')

        convs = [mpl.patches.Annulus((0,0), c, 0.0001, color='firebrick') for c in convection if c != 0]

        # xr = np.linspace(0, radius, 1000)
        # conv = sp.interpolate.interp1d(10**p.logR, p.brunt_N <= 0, fill_value = np.nan, bounds_error=0)(xr)
        # print(conv)
        
        # conv = np.array([sp.interpolate.interp1d(10**p.logR, p.brunt_N<0, 
        #             fill_value=np.nan, bounds_error=0)(xr) 
        #         for p in self.profs])
        
        ppcnomin = np.min(10**p.logR[(p.pp+p.cno) > 0.01]) 
        ppcnomax = np.max(10**p.logR[(p.pp+p.cno) > 0.01])
        triamin = np.min(10**p.logR[(p.tri_alpha) > 0.01])
        triamax = np.max(10**p.logR[(p.tri_alpha) > 0.01])
            
        h = mpl.patches.Annulus((0,0), ppcnomax, ppcnomax-ppcnomin, hatch='\\\\', alpha=0)
        he = mpl.patches.Annulus((0,0), triamax, triamax-triamin, hatch='////', alpha=0)

        surface = mpl.patches.Annulus((0,0), radius + 0.1, 0.05, color=surface_color)


        plt.gca().add_patch(base)
        # plt.gca().add_patch(conv2_top)
        # plt.gca().add_patch(conv2_bot)
            
        # plt.gca().add_patch(conv1_top)
        # plt.gca().add_patch(conv1_bot)

        # plt.gca().add_patch(conv_core)
        for conv_patch in convs:
            plt.gca().add_patch(conv_patch)
        # plt.scatter(np.linspace(0, 0, len(convection)), convection, color='firebrick')

        # plt.gca().add_patch(conv)

        plt.gca().add_patch(h)
        plt.gca().add_patch(he)

        plt.gca().add_patch(surface)

        
        # if age_plot is True:
        #     plt.text(x=0.8, y=0.95, s=f'Age: {10**-9*self.get_value_profile('star_age', profile_num):.2f} Gyr',
        #              transform=plt.gca().transAxes)
            
            
        if age_plot is True:
            plt.text(x=0.8, y=0.95, s=f'Age: {age_scale*self.get_value_profile('star_age', profile_num):.2f} {age_label}',
                     transform=plt.gca().transAxes)
            
        if xlim is None:
            xlim=[-max_r, max_r]
            

        plt.xlim(xlim)
        plt.ylim(xlim)
        plt.xticks([])
        plt.yticks([])
        # plt.legend()
