######## GRL : Fig 3  ##################
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
import pandas as pd
import seaborn as sns  
import matplotlib.colors as mcolors

######### Draw subplots ##############################
fig, ax = plt.subplots(1, 3, figsize = (18,5), dpi = 400) 

########### Fig d,e,f Heatmap###############################
N_hour_plot = 4 
x_axis = np.arange(0, N_hour_plot*60+1, 1) 
y_axis = np.arange(0, 60 +1/4., 1/4. ) 
f_max =   0.5 
f_min = - f_max

def heatmap_sns(episode, Avg_df, v_min, v_max):
    N_hour_plot = 4 
    x_axis = np.arange(0, N_hour_plot*60+1, 1)  
    y_axis = np.arange(0, 60 +1/4., 1/4. ) 
    cmap_cus = sns.diverging_palette(240, 10, sep=20, as_cmap=True) 
    df = pd.DataFrame(data=Avg_df, index=y_axis, columns=x_axis/60.)
    mask = np.isnan(df)
    ax = sns.heatmap(df, cmap=cmap_cus, vmin=v_min, vmax=v_max, xticklabels=30, yticklabels=10, mask=mask, cbar_kws={'extend': 'both', 'label': 'QV anomaly [g/kg]'})
    ax.set_facecolor((0.95, 0.9, 0.7, 0.3)) 
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='gray')
    ax.invert_yaxis()
    ax.set_ylabel('Radius [km]')
    ax.set_xlabel('Age [hour]')

ax[0].set_title("Episode 1 (suppressed)")
ax[1].set_title("Episode 2 (growing)")
ax[2].set_title("Episode 3 (enhanced)")

####################   Remove noises #############################################
cdf_thsh_same = 1e-4

def remove_noise(Avg_df, PDFn_df, cdf_thsh):
    CDF_df = np.zeros((len(PDFn_df[:,0]), len(PDFn_df[0,:])), float)
    for ix in range(len(PDFn_df[0,:])):
        col_ix = PDFn_df[:,ix]
        cdf_ix = np.nancumsum(col_ix) / np.nansum(col_ix)   
        Avg_df[:,ix]  = np.where(cdf_ix <= (1-cdf_thsh), Avg_df[:,ix], np.nan)
        CDF_df[:,ix]  = cdf_ix  
    return Avg_df, CDF_df

########### Fig 3d: heatmap in episode 1 #####################################
plt.subplot(1,3,1) 
dir_d = './data/Fig3/'
Avg_df_d  = np.load(dir_d + 'E1/QVano-ageMax4hr-220925-e1.npy')
PDFn_df_d = np.load(dir_d + 'E1/PDF-qvANO-ageMax4hr-220925-e1.npy')   # (radius, age)
Avg_df_d, CDF_df_d = remove_noise(Avg_df_d, PDFn_df_d, cdf_thsh_same)
heatmap_sns(1, Avg_df_d, f_min, f_max)

########### Fig 3e: heatmap in episode 2 #####################################
plt.subplot(1,3,2)
dir_e = dir_d
Avg_df_e  = np.load(dir_e + 'E2/QVano-ageMax4hr-220925-e2.npy')
PDFn_df_e = np.load(dir_e + 'E2/PDF-qvANO-ageMax4hr-220925-e2.npy')
Avg_df_e, CDF_df_e = remove_noise(Avg_df_e, PDFn_df_e, cdf_thsh_same)
heatmap_sns(2, Avg_df_e, f_min, f_max)

########### Fig 3f: heatmap in episode 3 #####################################
plt.subplot(1,3,3) 
dir_f = dir_d
Avg_df_f  = np.load(dir_f + 'E3/QVano-ageMax4hr-220925-e3.npy')
PDFn_df_f  = np.load(dir_f + 'E3/PDF-qvANO-ageMax4hr-220925-e3.npy')
Avg_df_f, CDF_df_f = remove_noise(Avg_df_f, PDFn_df_f, cdf_thsh_same)
heatmap_sns(3, Avg_df_f, f_min, f_max)

##### Save Figures ##################################
ax[0].set_title('(a)',loc='left')
ax[1].set_title('(b)',loc='left')
ax[2].set_title('(c)',loc='left')

fig.tight_layout()
figname = './fig/8.0.Fig3-CP-QV_ANO_heatmaps-LPDM-removeQVnoise-240302'
plt.savefig(figname + '.png')
plt.savefig(figname + '.jpg')




