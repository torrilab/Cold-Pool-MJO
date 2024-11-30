######## GRL : Fig 1  ##################
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

def add_column1(array_8ph):
    arr_list = list(array_8ph)
    arr_column0 = arr_list[0]
    arr_list.append(arr_column0)
    array_9ph = np.array(arr_list)

    return array_9ph

fig, ax = plt.subplots(1, 2, figsize = (16,4), dpi = 400)

##############  Fig 1a ############################################################
#### original Fig 1b ################################
plt.subplot(1,2,1) 
dir_1a = './data/Fig1/'
dir_1b = dir_1a
dir_1b_Iorg = dir_1b + 'Fig1c-SAM_3kmsearch_Iorg-230206/'

x_1a_orig = range(1, 9, 1)
x_1a = range(1, 10, 1)
x_ticks = list(x_1a_orig)+ [1]
print('x_ticks', x_ticks)

y1_Iorg = np.load(dir_1b_Iorg + 'y1-25perc.npy')
y2_Iorg = np.load(dir_1b_Iorg + 'y2-50perc.npy')
y3_Iorg = np.load(dir_1b_Iorg + 'y3-75perc.npy')

y1_Iorg = add_column1(y1_Iorg)
y2_Iorg = add_column1(y2_Iorg)
y3_Iorg = add_column1(y3_Iorg)

plt.fill_between(x_1a, y1_Iorg, y3_Iorg, color='orange', alpha=0.2)
l1=plt.plot(x_1a,y1_Iorg,'k--',label='25 %',linewidth=0.5)
l2=plt.plot(x_1a,y2_Iorg,'k-.', label='50 %',linewidth=1)
l3=plt.plot(x_1a,y3_Iorg,'k--',label='75 %',linewidth=0.5)

dir_1c_PR = dir_1a + 'Fig1cd-SAM_PR-230204/'
y1_PR   = np.load(dir_1c_PR + 'y1-25perc.npy')
y2_PR   = np.load(dir_1c_PR+ 'y2-50perc.npy')
y3_PR   = np.load(dir_1c_PR + 'y3-75perc.npy')

y1_PR = add_column1(y1_PR)
y2_PR = add_column1(y2_PR)
y3_PR = add_column1(y3_PR)

axb2 = ax[0].twinx()
plt.fill_between(x_1a, y1_PR, y3_PR, color='blue', alpha=0.2)
l1=plt.plot(x_1a,y1_PR,'k--',label='25 %',linewidth=0.5)
l2=plt.plot(x_1a,y2_PR,'k-.', label='50 %',linewidth=1)
l3=plt.plot(x_1a,y3_PR,'k--',label='75 %',linewidth=0.5)

ax[0].set_title('I_org and PR (3km search distance)')
ax[0].set_xlabel('MJO Local Phase ')
ax[0].set_ylabel('I_org')
axb2.set_ylabel('Precipitation Rate [mm/hr]')
ax[0].legend(loc=1)


##############  Fig 1b ############################################################
#### original Fig 1d ################################
plt.subplot(1,2,2) 
dir_1d = dir_1a
dir_1d_Nc = dir_1d + 'Fig1d-SAM_Nc_3km-230726/'

y1_Nc = np.load(dir_1d_Nc + 'y1-25perc.npy')
y2_Nc = np.load(dir_1d_Nc + 'y2-50perc.npy')
y3_Nc = np.load(dir_1d_Nc + 'y3-75perc.npy')

y1_Nc_9 = add_column1(y1_Nc)
y2_Nc_9 = add_column1(y2_Nc)
y3_Nc_9 = add_column1(y3_Nc)

plt.fill_between(x_1a, y1_Nc_9, y3_Nc_9, color='green', alpha=0.2)
l1=plt.plot(x_1a,y1_Nc_9,'k--',label='25 %',linewidth=0.5)
l2=plt.plot(x_1a,y2_Nc_9,'k-.', label='50 %',linewidth=1)
l3=plt.plot(x_1a,y3_Nc_9,'k--',label='75 %',linewidth=0.5)

ax[1].set_title('Nc (3km search distance)')
ax[1].set_xlabel('MJO Local Phase ')
ax[1].set_ylabel('Number of convective clusters')
ax[1].legend(loc=1)


##### Save Figures ####################################################
ax[0].set_title('(a)',loc='left')
ax[1].set_title('(b)',loc='left')

lw_ax    = 0.3
alpha_ax = 0.5
ax[0].grid(ls='--', lw=lw_ax, color='gray', alpha=alpha_ax)
ax[1].grid(ls='--', lw=lw_ax, color='gray', alpha=alpha_ax)
ax[0].set_xticks(x_ticks)
ax[1].set_xticks(x_ticks)
ax[0].set_xlim(1, 8.5)
ax[1].set_xlim(1, 8.5)

fig.tight_layout()
plt.savefig('./fig/Fig1-7.5.png')
plt.savefig('./fig/Fig1-7.5.jpg')

















