######## GRL : Fig 2  ##################
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
import pandas as pd
import seaborn as sns  
from matplotlib.gridspec import GridSpec

######### Draw subplots ##############################
fig, axs = plt.subplots(1, 3, figsize = (6*3, 5), dpi = 400) 
axs = axs.flatten()
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

plt.subplot(1,3,1)
dir_1a = './data/Fig2/'
PDFn_grid_ph1 = np.loadtxt(dir_1a + "Heatmap-PhaseColumnPartial-D-orders-8phases-1122/max20/Phase1-heatmap-Partial-Order_10-D_20.txt")  
PDFn_grid_ph2 = np.loadtxt(dir_1a + "Heatmap-PhaseColumnPartial-D-orders-8phases-1122/max20/Phase2-heatmap-Partial-Order_10-D_20.txt")
PDFn_grid_ph5 = np.loadtxt(dir_1a + "Heatmap-PhaseColumnPartial-D-orders-8phases-1122/max20/Phase5-heatmap-Partial-Order_10-D_20.txt")
PDFn_grid_ph6 = np.loadtxt(dir_1a + "Heatmap-PhaseColumnPartial-D-orders-8phases-1122/max20/Phase6-heatmap-Partial-Order_10-D_20.txt")
Diff_ph56_12  = ( PDFn_grid_ph5 + PDFn_grid_ph6 - PDFn_grid_ph1 -PDFn_grid_ph2 )
### Draw heatmaps #########
Max_order = 10
Max_order_inrange = Max_order
Order_range_plot = range(0, Max_order_inrange, 1)
D_max = 20
D_binsedge = range(1, D_max+1, 1) 
x_axis  = Order_range_plot
y_axis = D_binsedge
df = pd.DataFrame(data = Diff_ph56_12, index = y_axis, columns = x_axis)
ax1 = sns.heatmap(df, linewidth=0.5, vmin=-4, vmax=4, cmap="RdBu_r")
ax1.invert_yaxis() 
ax1.set_title('Percentage [%]')
ax1.set_ylabel('Actual Distance [km]')
ax1.set_xlabel('Order of Neighbor Distances (0:nearest)')

#### def for PDF, 8 lines colored by phases ; for fig 2b & 2c #######################
phase_list = range(1, 9, 1)
cmap12345 = plt.get_cmap('seismic')
colors12345 = cmap12345(np.linspace(0.1, 1, num = len(phase_list)//2 +1 ))
cmap56781 = plt.get_cmap('seismic')  
colors56781 = cmap56781(np.linspace(1, 0.1, num = len(phase_list)//2 +1 ))
colors = []
linewids = [] 
markers = []
for item in colors12345[:-1]: 
    colors.append(item)
    linewids.append(0.5)  
    markers.append('^--')
for item in colors56781[:-1]: 
    colors.append(item)
    linewids.append(1.0)
    markers.append('v-')

##### Fig 2b: PDFs of clustered updrafts' sizes #################################
plt.subplot(1,3,2)
dir_1b = dir_1a + 'PDF-size_updraftclu-8lines-8ph-230422/'
X_2b = np.load(dir_1b + 'X_size-8ph-230422.npy')
Y_2b = np.load(dir_1b + 'Y_perc-8ph-230422.npy')

for il in range(0, len(phase_list), 1):
    l_ip=plt.plot(X_2b[il], Y_2b[il], c = colors[il], label='Ph'+str( phase_list[il] ), linewidth= linewids[il]*2.) 

ax2.set_title('Histogram of all sizes')
ax2.set_ylabel('Probability')
ax2.set_xlabel('Sizes of updraft clusters [km^2]')
ax2.legend(loc = 1) 

##### Fig 2c: PDFs of all distances between all clusters 
plt.subplot(1,3,3)
dir_2c = dir_1a + 'Fig2c-PDF-9lines-D_all-8ph_1random/'
X_2c = np.load(dir_2c + 'X-8PH-allDis.npy', allow_pickle=True)
Y_2c = np.load(dir_2c + 'Y-8PH-prob.npy', allow_pickle=True)
x_r   = np.load(dir_2c + 'X-random-allDis.npy', allow_pickle=True)  # randomly distributed 10000 points
y_r   = np.load(dir_2c + 'Y-random-allDis.npy', allow_pickle=True)

for il in range(0, len(phase_list), 1):
    l_ip=plt.plot(X_2c[il], Y_2c[il], c = colors[il], label='Ph'+str( phase_list[il] ), linewidth= linewids[il]*2.)

l_random = plt.plot(x_r, y_r, c = 'k', label='Ran', linewidth= 1)   

ax3.set_title('Histogram of all distances')
ax3.set_ylabel('Probability')
ax3.set_xlabel('Distances between updraft clusters [km]')
ax3.legend(loc = 1)

##### Save Figures ##################################
ax1.set_title('(a)',loc='left')
ax2.set_title('(b)',loc='left')
ax3.set_title('(c)',loc='left')
plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.1, top=0.9, left=0.1, right=0.9)

ax2.grid(ls='--', lw=0.2, color='gray', alpha=0.5)
ax3.grid(ls='--', lw=0.2, color='gray', alpha=0.5)

fig.tight_layout()
figname = './fig/8.3.Fig2-Distances-1km-lines-c_up-231218'
plt.savefig(figname + '.png')
plt.savefig(figname + '.jpg')














