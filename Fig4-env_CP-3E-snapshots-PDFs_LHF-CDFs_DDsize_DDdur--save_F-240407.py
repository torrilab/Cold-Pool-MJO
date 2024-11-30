######## GRL : Fig 4  ##################
### Mingyue Tang, 04/07/2024 ###########

import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib   
matplotlib.use('Agg')  
from matplotlib import pyplot as plt  
import glob
import torch
from tqdm import tqdm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

addlabel_list = [' (suppressed)', ' (growing)', ' (enhanced)']

######### Draw subplots ##############################
fig = plt.figure(figsize=(18, 8.5), dpi = 400)   #12, 10; 18,12; 
gs = GridSpec(2, 2*3, figure=fig, height_ratios=[1, 2./3.], width_ratios=[1]*6)

###### No saved screenshot results for Fig 4a, b, c ##########################
def snapshot(ax, data, it, vmin, vmax, fig, addlabel_list_i, f_save):
    ncfile = xr.open_dataset(data)
    f = ncfile["LHF"] #"Latent Heat Flux", W/m2
    ncfile.close()
    x_axis = f.x / 250. 
    y_axis = f.y / 250.    
    f = np.where(f > vmax, vmax+1, f) 
    f = np.where(f < vmin, vmin-1, f)
    color_levels = 10
    norm2 = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ticks = np.linspace(vmin, vmax, color_levels+1)
    print('ticks', ticks)

    C = ax.contourf(x_axis, y_axis, f, cmap="viridis", levels = color_levels,\
                    vmin=vmin, vmax=vmax, norm=norm2, \
                        extend='both'
                    )

    ax.set_ylabel('x [km]')  
    ax.set_xlabel('y [km]')
    ax.set_title('time step: ' + str(it) + addlabel_list_i)
    colorbar = fig.colorbar(C, ax=ax, orientation='vertical',
                            extend='both', ticks=ticks ,
                            label='Latent heat fluxes [W/m^2]',  
                            )  

######### Fig 4a ####################################
ax1 = fig.add_subplot(gs[0, 0:2])
data_a = './data/Fig4/4abc/E1-LHF.npy'
it_slc_a = 83  #### Time selected ##
vmin, vmax = 80, 180
snapshot1 = snapshot(ax1, data_a, it_slc_a, vmin, vmax, fig, addlabel_list[0])

######### Fig 4b ####################################
ax2 = fig.add_subplot(gs[0, 2:4])
data_b = './data/Fig4/4abc/E2-LHF.npy'
it_slc_b = 220  #### Time selected ##
vmin, vmax = 80, 180
snapshot2 = snapshot(ax2, data_b, it_slc_b, vmin, vmax, fig, addlabel_list[1])

######### Fig 4c ####################################
ax3 = fig.add_subplot(gs[0, 4:6])
data_c = './data/Fig4/4abc/E3-LHF.npy'
it_slc_c = 83 #### Time selected ##
vmin, vmax = 80, 180
snapshot2 = snapshot(ax3, data_c, it_slc_c, vmin, vmax, fig, addlabel_list[2])

###### Settings in Fig d, e, f: PDFs and CDFs lines #####
colors = ['blue', 'black', 'red']
line_width = 1.3

######### Fig 4d, new added, 08/10/2023 ###############
#### PDFs of LHF at every grid box at every time step, compare 3 episodes.
ax4 = fig.add_subplot(gs[1, 0:2])
dir_d = './data/Fig4/4d/'
X_d = np.load(dir_d + 'X-NND-CDFs-xran_250_25-230806.npy')
Y_d = np.load(dir_d + 'Y-NND-per-CDFs-xran_250_25-230806.npy')
for ie in range(3):
    l_ie = plt.plot(X_d[ie], Y_d[ie], c = colors[ie],\
                    label='Episode '+str(ie+1)+ addlabel_list[ie], linewidth=line_width)
ax4.set_title("PDFs of LHF")
ax4.set_xlabel('Latent Heat Fluxes [W/m^2]]')
ax4.set_ylabel('Percentile [%]')
ax4.legend(loc='upper right')
ax4.grid(ls='--', lw=0.5, color='gray', alpha=0.5)

######## Fig 4e ############################################
### CDFs of DD max size ####################################
ax5 = fig.add_subplot(gs[1, 2:4])
path_e = './data/Fig4/4e/'
X_size = np.load(path_e + 'X-km2-Size-CDFs-xmax_288-230424.npy', allow_pickle=True)  # unit: km, /16 already.
Y_size = np.load(path_e + 'Y_maxsize-per-CDFs-xmax_288-230403.npy', allow_pickle=True)
for ie in range(3):
    l_ie = plt.plot(X_size[ie], Y_size[ie]*100., c = colors[ie],\
                    label='Episode '+str(ie+1)+ addlabel_list[ie], linewidth=line_width)
ax5.set_title("CDFs of coreID's max size at 12 m")
ax5.set_xlabel('Maximum size [km^2]')
ax5.set_ylabel('Percentile [%]')
ax5.legend(loc='lower right')
ax5.grid(ls='--', lw=0.5, color='gray', alpha=0.5)

######### Fig 4f #############################################
ax6 = fig.add_subplot(gs[1, 4:6])
path_f = './data/Fig4/4f/'
X_dura = np.load(path_e + 'X-dura-CDFs-xmax_125-230403.npy', allow_pickle=True)  # unit: min
Y_dura = np.load(path_e + 'Y_dura-per-CDFs-xmax_125-230403.npy', allow_pickle=True) 
for ie in range(3):
    l_ie = plt.plot(X_dura[ie], Y_dura[ie]*100., c = colors[ie],\
                    label='Episode '+str(ie+1)+ addlabel_list[ie], linewidth=line_width)
ax6.set_title("CDFs of coreID's duration")
ax6.set_xlabel('Duration [min]')
ax6.set_ylabel('Percentile [%]')
ax6.legend(loc='lower right')
ax6.grid(ls='--', lw=0.5, color='gray', alpha=0.5)

##### Save Figures #####################################
ax1.set_title('(a)',loc='left')
ax2.set_title('(b)',loc='left')
ax3.set_title('(c)',loc='left')
ax4.set_title('(d)',loc='left')
ax5.set_title('(e)',loc='left')
ax6.set_title('(f)',loc='left')

fig.tight_layout()
figname = './fig/3.0.Fig4-LHFsnap-PDF_LHF-CDFcoreID-240302'
plt.savefig(figname + '.png')
plt.savefig(figname + '.jpg')









