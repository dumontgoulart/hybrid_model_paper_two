# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:39:47 2021

@author: morenodu
"""
import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')
# from mask_shape_border import mask_shape_border
from failure_probability import feature_importance_selection
from sklearnex import patch_sklearn
patch_sklearn()
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

#%% LIST OF FUNCTIONS

countries = shpreader.natural_earth(resolution='50m',category='cultural',name='admin_0_countries')
# Find the boundary polygon.
for country in shpreader.Reader(countries).records():
    if country.attributes['SU_A3'] == 'ARG':
        arg_border0 = country.geometry
    elif country.attributes['SU_A3'] == 'BRA':
        bra_border0 = country.geometry
    elif country.attributes['SU_A3'] == 'USA':
        usa_border0 = country.geometry

# Function for state mask and mapping
def states_mask(input_gdp_shp, state_names = None) :
    country = gpd.read_file(input_gdp_shp, crs="epsg:4326") 
    country_shapes = list(shpreader.Reader(input_gdp_shp).geometries())
    if state_names is not None:
        soy_states = country[country['NAME_1'].isin(state_names)]
        states_area = soy_states['geometry'].to_crs({'proj':'cea'}) 
        states_area_sum = (sum(states_area.area / 10**6))
        return soy_states, country_shapes, states_area_sum
    else:
        return country_shapes

def plot_2d_am_map(dataarray_2d, title = None, colormap = None, vmin = None, vmax = None):
    # Plot 2D map of DataArray, remember to average along time or select one temporal interval
    plt.figure(figsize=(12,10)) #plot clusters
    ax=plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-74.5, central_latitude=0))
    ax.add_feature(cartopy.feature.LAND, facecolor='gray', alpha=0.1)
    ax.add_geometries([usa_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
    ax.add_geometries([bra_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
    ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.0, zorder=0)
    if colormap is None:
        dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, vmin = vmin, vmax = vmax, zorder=20)
    elif colormap is not None:
        dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap= colormap, vmin = vmin, vmax = vmax, zorder=20)
    ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='None', alpha=0.5, lw=0.7, zorder=21)
    ax.set_extent([-115,-34,-41,44])
    if title is not None:
        plt.title(title)
    plt.show()
    
def plot_2d_map(dataarray_2d):
    # Plot 2D map of DataArray, remember to average along time or select one temporal interval
    plt.figure(figsize=(12,5)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
    ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-80.73,-34,-45,6], ccrs.PlateCarree())
    plt.show()

def plot_2d_us_map(dataarray_2d):
    # Plot 2D map of DataArray, remember to average along time or select one temporal interval
    plt.figure(figsize=(12,5)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
    ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-125,-67,25,50], ccrs.Geodetic())
    plt.show()

# Detrend Dataset
def detrend_dataset(DS, deg = 'free', dim = 'time', print_res = True, mean_data = None):
            
    if deg == 'free':
        da_list = []
        for feature in list(DS.keys()):
            da = DS[feature]
            print(feature)
            
            if mean_data is None:
                mean_dataarray = da.mean('time')
            else:
                mean_dataarray = mean_data[feature].mean('time') #da.mean('time') - ( da.mean() - mean_data[feature].mean() )
            
            da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )
    
            dict_res = {}
            for degree in [1,2]:
                # detrend along a single dimension
                p = da.polyfit(dim=dim, deg=degree)
                fit = xr.polyval(da[dim], p.polyfit_coefficients)
                
                da_det = da - fit
                
                res_detrend = np.nansum((da_zero_mean.mean(['lat','lon'])-da_det.mean(['lat','lon']))**2)
                dict_res.update({degree:res_detrend})
            if print_res == True:
                print(dict_res)
            deg = min(dict_res, key=dict_res.get) # minimum degree   
            
            # detrend along a single dimension
            print('Chosen degree is ', deg)
            p = da.polyfit(dim=dim, deg=deg)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
        
            da_det = da - fit + mean_dataarray
            da_det.name = feature
            da_list.append(da_det)
        DS_det = xr.merge(da_list) 
    
    else:       
        px= DS.polyfit(dim='time', deg=deg)
        fitx = xr.polyval(DS['time'], px)
        dict_name = dict(zip(list(fitx.keys()), list(DS.keys())))
        fitx = fitx.rename(dict_name)
        DS_det  = (DS - fitx) + mean_data
        
    return DS_det


# Different ways to detrend, select the best one
def detrend_dim(da, dim, deg = 'free', print_res = True):        
    if deg == 'free':
        
        da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )

        dict_res = {}
        for degree in [1,2]:
            # detrend along a single dimension
            p = da.polyfit(dim=dim, deg=degree)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
            
            da_det = da - fit
            res_detrend = np.nansum((da_zero_mean.mean(['lat','lon']) - da_det.mean(['lat','lon']))**2)
            dict_res_in = {degree:res_detrend}
            dict_res.update(dict_res_in)
        if print_res == True:
            print(dict_res)
        deg = min(dict_res, key=dict_res.get) # minimum degree        
    
    # detrend along a single dimension
    print('Chosen degree is ', deg)
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    
    da_det = da - fit   
    return da_det

def rearrange_latlot(DS, resolution = 0.5):
    DS = DS.sortby('lat')
    DS = DS.sortby('lon')
    new_lat = np.arange(DS.lat.min(), DS.lat.max() + resolution, resolution)
    new_lon = np.arange(DS.lon.min(), DS.lon.max() + resolution, resolution)
    DS = DS.reindex({'lat':new_lat})
    DS = DS.reindex({'lon':new_lon})
    return DS

def timedelta_to_int(DS, var):
    da_timedelta = DS[var].dt.days
    da_timedelta = da_timedelta.rename(var)
    da_timedelta.attrs["units"] = 'days'
    
    return da_timedelta

def weighted_conversion(DS, DS_area, name_ds = 'Yield'):
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) ) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])).to_dataset(name = name_ds)
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])))
    return DS_weighted.sum(['lat','lon'])

#%% LOADING MAIN DATA
# Load Country shapes
us1_shapes = states_mask('../../Paper_drought/data/gadm36_USA_1.shp')
br1_shapes = states_mask('../../Paper_drought/data/gadm36_BRA_1.shp')
arg_shapes = states_mask('GIS/gadm36_ARG_1.shp')

# =============================================================================
# USe MIRCA to isolate the rainfed 90% soybeans
# =============================================================================
# DS_mirca_test = xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).rename({'latitude': 'lat', 'longitude': 'lon','annual_area_harvested_rfc_crop08_ha_30mn':'harvest_area'})
DS_mirca_test = xr.open_dataset("../../paper_hybrid_agri/data/soy_harvest_spam_native_05x05.nc", decode_times=False)

#### HARVEST DATA
DS_harvest_area_sim = xr.load_dataset("../../paper_hybrid_agri/data/soybean_harvest_area_calculated_americas_hg.nc", decode_times=False)
DS_harvest_area_sim = DS_harvest_area_sim.sel(time = slice(2013,2015)).mean('time') #.sel(time = slice(1979,2016))
DS_harvest_area_sim = DS_harvest_area_sim.where(DS_mirca_test['harvest_area'] > 0 )
DS_mirca_test = DS_harvest_area_sim
# plot_2d_am_map(DS_mirca_test['harvest_area'].sel(time = 1980))
# plot_2d_am_map(DS_mirca_test['harvest_area'].sel(time = 2016))


# =============================================================================
# EPIC 
# =============================================================================
DS_y_epic = xr.open_dataset("epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc", decode_times=False)
# Convert time unit
units, reference_date = DS_y_epic.time.attrs['units'].split('since')
DS_y_epic['time'] = pd.date_range(start=reference_date, periods=DS_y_epic.sizes['time'], freq='YS')
DS_y_epic['time'] = DS_y_epic['time'].dt.year 
DS_y_epic = DS_y_epic.sel(time=slice('1972-12-12','2016-12-12'))
DS_y_epic = DS_y_epic.rename({'yield-soy-noirr':'yield'})

DS_y_epic_am = DS_y_epic.where(DS_mirca_test['harvest_area'] > 0 )
plot_2d_am_map(DS_y_epic_am['yield'].sel(time=1980))
plot_2d_am_map(DS_y_epic_am['yield'].sel(time=2016))


# =============================================================================
# # US -------------------------------------------------------------
# =============================================================================
DS_y_obs_us_all = xr.open_dataset("../../paper_hybrid_agri/data/soy_yields_US_all_1975_2020_05x05.nc", decode_times=False).sel(lon=slice(-160,-10))
# Convert time unit
units, reference_date = DS_y_obs_us_all.time.attrs['units'].split('since')
DS_y_obs_us_all['time'] = pd.date_range(start=reference_date, periods=DS_y_obs_us_all.sizes['time'], freq='YS').year
DS_y_obs_us_all = DS_y_obs_us_all.sel(time=slice('1975','2016'))

DS_y_obs_us_all = DS_y_obs_us_all.where(DS_mirca_test['harvest_area'] > 0 )
DS_y_obs_us_all.to_netcdf("soybean_yields_US_1978_2016.nc")
plot_2d_am_map(DS_y_obs_us_all['Yield'].mean('time'))

DS_y_epic_us = DS_y_epic_am.where(DS_y_obs_us_all['Yield'] > - 5)

# =============================================================================
# # BRAZIL --------------------------------------------------------
# =============================================================================
DS_y_obs_br = xr.open_dataset("../../paper_hybrid_agri/data/soy_yield_1975_2016_05x05_1prc.nc", decode_times=False) 
DS_y_obs_br=DS_y_obs_br.sel(time = slice('1975', '2016'))
DS_y_obs_br = DS_y_obs_br.where(DS_mirca_test['harvest_area'] > 0 )
plot_2d_am_map(DS_y_obs_br['Yield'].mean('time'))

DS_y_obs_br.to_netcdf("soybean_yields_BR_1978_2016.nc")

# SHIFT EPIC FOR BRAZIL ONE YEAR FORWARD TO MATCH INTERNATIONAL CALENDARS
DS_y_epic_br = DS_y_epic_am.where(DS_y_obs_br['Yield'].mean('time') > - 5)
DS_y_epic_br = DS_y_epic_br.copy().shift(time = 1) # SHIFT EPIC BR ONE YEAR FORWARD
DS_y_epic_br = DS_y_epic_br.where(DS_y_obs_br['Yield'] > - 5)
# plot_2d_am_map(DS_y_epic_br['yield'].sel(time=1979))

# =============================================================================
# # AREGNTINA --------------------------------------------------------
# =============================================================================
DS_y_obs_arg = xr.open_dataset("../../paper_hybrid_agri/data/soy_yield_arg_1974_2019_05x05.nc", decode_times=False)#soy_yield_1980_2016_1prc05x05 / soy_yield_1980_2016_all_filters05x05
# DS_y_obs_arg=DS_y_obs_arg.sel(time = slice('1978', '2016'))

# SHIFT OBSERVED DATA FOR ARGENTINA ONE YEAR FORWARD TO MATCH INTERNATIONAL CALENDARS - from planting year to harvest year
DS_y_obs_arg = DS_y_obs_arg.copy().shift(time = 1) # SHIFT AGRNEITNA ONE YeAR FORWARD
DS_y_obs_arg = DS_y_obs_arg.where(DS_mirca_test['harvest_area'] > 0 )
DS_y_obs_arg.to_netcdf("soybean_yields_ARG_1978_2016.nc")
plot_2d_am_map(DS_y_obs_arg['Yield'].sel(time=1985))

# SHIFT EPIC DATA FOR ARGENTINA ONE YEAR FORWARD TO MATCH INTERNATIONAL CALENDARS
DS_y_epic_arg = DS_y_epic_am.where(DS_y_obs_arg['Yield'].mean('time') > - 5)
DS_y_epic_arg = DS_y_epic_arg.copy().shift(time = 1) # SHIFT EPIC ARG ONE YeAR FORWARD
DS_y_epic_arg = DS_y_epic_arg.where(DS_y_obs_arg['Yield'] > - 5)

# =============================================================================
# # Plots for analysis
# =============================================================================
plt.plot(DS_y_obs_arg['Yield'].time, DS_y_obs_arg['Yield'].mean(['lat','lon']), label = 'ARG')
plt.plot(DS_y_obs_br.time, DS_y_obs_br['Yield'].mean(['lat','lon']), label = 'BR')
plt.plot(DS_y_obs_us_all.time, DS_y_obs_us_all['Yield'].mean(['lat','lon']), label = 'US')
plt.vlines(DS_y_obs_us_all.time, 1,3.5, linestyles ='dashed', colors = 'k')
plt.legend()
plt.show()

plt.plot(DS_y_epic_arg.time, DS_y_epic_arg['yield'].mean(['lat','lon']), label = 'ARG')
plt.plot(DS_y_epic_br.time, DS_y_epic_br['yield'].mean(['lat','lon']), label = 'BR')
plt.plot(DS_y_epic_us.time, DS_y_epic_us['yield'].mean(['lat','lon']), label = 'US')
plt.vlines(DS_y_epic_us.time, 2,4.5, linestyles ='dashed', colors = 'k')
plt.legend()
plt.show()

# =============================================================================
# # Combine the datsaets:
# =============================================================================
DS_y_obs_am = DS_y_obs_us_all.combine_first(DS_y_obs_arg)
DS_y_obs_am = DS_y_obs_am.combine_first(DS_y_obs_br)
DS_y_obs_am = rearrange_latlot(DS_y_obs_am)

plot_2d_am_map(DS_y_obs_am["Yield"].mean('time'))

DS_y_epic_am_2 = DS_y_epic_us.combine_first(DS_y_epic_br)
DS_y_epic_am_2 = DS_y_epic_am_2.combine_first(DS_y_epic_arg)

plt.plot(DS_y_epic_am.time, DS_y_epic_am['yield'].mean(['lat','lon']), label = "no shift")
plt.plot(DS_y_epic_am_2.time, DS_y_epic_am_2['yield'].mean(['lat','lon']), label = "shift")
plt.legend()
plt.show()

DS_y_epic_am = DS_y_epic_am_2.copy()

plot_2d_am_map(DS_y_epic_am["yield"].mean('time'))
plot_2d_am_map(DS_y_obs_am["Yield"].mean('time'))

##### concatenate the two types of data
DS_y_obs_am['Yield'] = DS_y_obs_am['Yield'].where(DS_y_epic_am['yield'] >= 0.0 )
DS_y_epic_am['yield'] = DS_y_epic_am['yield'].where(DS_y_obs_am['Yield'] >= 0.0 )

corr_3d = xr.corr(DS_y_epic_am["yield"], DS_y_obs_am["Yield"], dim="time", )
plot_2d_am_map(corr_3d)

corr_3d_high = corr_3d.where(corr_3d > 0.4)
plot_2d_am_map(corr_3d_high)

# Compare
df_epic_am = DS_y_obs_am.to_dataframe().dropna()
df_obs_am = DS_y_obs_am.to_dataframe().dropna()

# Detrend timeseries
DS_y_obs_am_det = xr.DataArray( detrend_dim(DS_y_obs_am['Yield'], 'time') + DS_y_obs_am['Yield'].mean('time'), name= DS_y_obs_am['Yield'].name, attrs = DS_y_obs_am['Yield'].attrs)
DS_y_obs_am_det = DS_y_obs_am_det.sel(time = slice(1979,2016))
DS_y_obs_am_det.to_netcdf("soybean_yields_america_detrended_1978_2016.nc")

DS_y_epic_am_det = xr.DataArray( detrend_dim(DS_y_epic_am["yield"], 'time') + DS_y_epic_am["yield"].mean('time'), name= DS_y_epic_am["yield"].name, attrs = DS_y_epic_am["yield"].attrs)
DS_y_epic_am_det = DS_y_epic_am_det.sel(time = slice(1979,2016))

plt.vlines(DS_y_epic_am_det.time, np.min(DS_y_obs_am_det.mean(['lat','lon'])), np.max(DS_y_epic_am_det.mean(['lat','lon'])), linestyles ='dashed', colors = 'k')
plt.plot(DS_y_epic_am_det.time, DS_y_epic_am_det.mean(['lat','lon']), label='EPIC')
plt.plot(DS_y_obs_am_det.time, DS_y_obs_am_det.mean(['lat','lon']), label='Observed')
plt.legend()
plt.show()

# Compare EPIC with Observed dataset
df_epic_am_det = DS_y_epic_am_det.to_dataframe().dropna().sort_index(ascending = [True,False,True])
df_obs_am_det = DS_y_obs_am_det.to_dataframe().dropna().sort_index(ascending = [True,False,True])

# Pearson's correlation
from scipy.stats import pearsonr

corr_grid, _ = pearsonr(df_obs_am_det['Yield'], df_epic_am_det['yield'])
print('Pearsons correlation at grid level: %.3f' % corr_grid)

corr_grouped, _ = pearsonr(df_obs_am_det['Yield'].groupby('time').mean(), df_epic_am_det['yield'].groupby('time').mean())
print('Pearsons correlation at aggregated level: %.3f' % corr_grouped)

# Plot each country detrended and weighted
DS_y_obs_det_weighted_am = weighted_conversion(DS_y_obs_am_det.sel(time = slice(1975,2016)), DS_area = DS_mirca_test.where(DS_y_obs_am['Yield'] > -1))
DS_y_obs_det_weighted_us = weighted_conversion(DS_y_obs_am_det, DS_area = DS_mirca_test.where(DS_y_obs_us_all['Yield'] > -1))
DS_y_obs_det_weighted_br = weighted_conversion(DS_y_obs_am_det, DS_area = DS_mirca_test.where(DS_y_obs_br['Yield'] > -1))
DS_y_obs_det_weighted_arg = weighted_conversion(DS_y_obs_am_det, DS_area = DS_mirca_test.where(DS_y_obs_arg['Yield'] > -1))

# Plot historical timeline of weighted soybean yield
plt.plot(DS_y_obs_det_weighted_us.time, DS_y_obs_det_weighted_us['Yield'], label = 'US')
plt.plot(DS_y_obs_det_weighted_br.time, DS_y_obs_det_weighted_br['Yield'], label = 'BR')
plt.plot(DS_y_obs_det_weighted_arg.time, DS_y_obs_det_weighted_arg['Yield'], label = 'ARG')
plt.legend()
plt.title('Weighted averages of soybean yields')
plt.ylabel('Yield (ton/ha)')
plt.tight_layout()
plt.show()

#%% Machine learning model training
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.layers import Activation
from keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout
from scikeras.wrappers import KerasRegressor
import lightgbm as lgb
from tensorflow.keras import regularizers

import os
os.environ['PYTHONHASHSEED']= '123'
os.environ['TF_CUDNN_DETERMINISTIC']= '1'
import random as python_random
np.random.seed(1)
python_random.seed(1)
tf.random.set_seed(1)

def calibration(X_origin,y_origin,type_of_model='RF'):
    # Shuffle and Split data
    X, y = shuffle(X_origin, y_origin, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
       
    if type_of_model == 'RF':
        model_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,
                                          max_depth = 20, max_features = 'auto',
                                          min_samples_leaf = 1, min_samples_split=2)
        
        full_model_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,
                                          max_depth = 20, max_features = 'auto',
                                          min_samples_leaf = 1, min_samples_split=2)
        
    elif type_of_model == 'lightgbm':
        model_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', lgb.LGBMRegressor(linear_tree= True, max_depth = 20, num_leaves = 50, min_data_in_leaf = 100, 
                                            random_state=0, learning_rate = 0.01, n_estimators = 1000 ) )
        ])
        
        
        full_model_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', lgb.LGBMRegressor(linear_tree= True, max_depth = 20, num_leaves = 50, min_data_in_leaf = 100, 
                                            random_state=0, learning_rate = 0.01, n_estimators = 1000 ) )
        ])
    
    elif type_of_model == 'DNN':
# 1024, 800, 512, 0.001, 0.2, 0, 433,
# 256	800	512	0.005	0.2, 442
# 1024	800	512	0.01	0.2	0	483
# 256 and epoch: 800 and neurons: 512 and learning rate : 0.01, dropout: 0.2,l2: 0 
             
        epochs_train = 529
        batch_size_train = 1024
        nodes_size = 512
        learning_rate_train = 0.01
        dropout_train = 0.2
        regul_value = 0
        # =============================================================================
        #      #   TRAIN model 
        # =============================================================================
        def create_model():
            train_model = Sequential()
            train_model.add(Dense(nodes_size, input_dim=len(X_train.columns), kernel_regularizer=regularizers.l2(regul_value))) 
            train_model.add(BatchNormalization())
            train_model.add(Activation('relu'))
            train_model.add(Dropout(dropout_train))
    
            train_model.add(Dense(nodes_size, kernel_regularizer=regularizers.l2(regul_value)))
            train_model.add(BatchNormalization())
            train_model.add(Activation('relu'))
            train_model.add(Dropout(dropout_train))
    
            train_model.add(Dense(nodes_size, kernel_regularizer=regularizers.l2(regul_value)))
            train_model.add(BatchNormalization())
            train_model.add(Activation('relu'))
            train_model.add(Dropout(dropout_train))
           
            train_model.add(Dense(nodes_size, kernel_regularizer=regularizers.l2(regul_value)))
            train_model.add(BatchNormalization())
            train_model.add(Activation('relu'))
            train_model.add(Dropout(dropout_train))
            
            train_model.add(Dense(nodes_size, kernel_regularizer=regularizers.l2(regul_value)))
            train_model.add(BatchNormalization())
            train_model.add(Activation('relu'))
            train_model.add(Dropout(dropout_train))
            
            train_model.add(Dense(1, activation='linear'))
            
            # compile the keras model
            train_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate_train), metrics=['mean_squared_error','mean_absolute_error'])
            return train_model
        
        # Callbacks to monitor the performance of the optimization of the model and if there is any overfitting
        # callback_model = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 100, restore_best_weights=True)
        # mc = ModelCheckpoint('best_model_test.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        
        model_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', KerasRegressor(model=create_model(), epochs= epochs_train, random_state = 0, batch_size=batch_size_train, verbose=0)) ]) #, callbacks=callback_model #, validation_split= 0.1, callbacks=[callback_model,mc]
        
        model_fit = model_rf.fit(X_train, y_train)
        
        # =============================================================================
        #         # Entire full set model
        # =============================================================================
        full_model_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', KerasRegressor(model=create_model(), epochs = epochs_train, random_state = 0, batch_size = batch_size_train, verbose=0)) ]) # validation_split= 0.1, callbacks=[callback_model_full, mc_full]
        
        model_fit_full = full_model_rf.fit(X, y)

        
    if type_of_model == 'DNN':
        # Wrap up and plot graphs
        model = model_fit
        
        full_model = model_fit_full
        
    else:
    
        model = model_rf.fit(X_train, y_train)
        
        full_model = full_model_rf.fit(X, y)
    
    # Test performance
    y_pred = model.predict(X_test)
    df_y_pred = pd.DataFrame(y_pred, index = y_test.index, columns = [y_test.name]) 
    
    # report performance
    print(f'Results for model: {type_of_model}')
    print("R2 on test set:", round(r2_score(y_test, y_pred),2))
    print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
    print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
    print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
    print("______")
    
    y_pred_total = full_model.predict(X)
    df_y_pred_total = pd.DataFrame(y_pred_total, index = y.index, columns = [y.name]) 
    df_y_pred_total = df_y_pred_total.sort_index()
    # Display error
    plt.figure(figsize=(5,5), dpi=250) #plot clusters
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color = 'black', label = '1:1 line')
    plt.ylabel('Predicted yield')
    plt.xlabel('Observed yield')
    plt.title('Scatter plot - test set')
    plt.legend()
    # plt.savefig('paper_figures/???.png', format='png', dpi=500)
    plt.show()
    
    # Display error
    plt.figure(figsize=(5,5), dpi=250) #plot clusters
    plt.scatter(y, y_pred_total)
    plt.plot(y, y, color = 'black', label = '1:1 line')
    plt.ylabel('Predicted yield')
    plt.xlabel('Observed yield')
    plt.title('Scatter plot - total set')
    plt.legend()
    # plt.savefig('paper_figures/???.png', format='png', dpi=500)
    plt.show()
   
    return df_y_pred, df_y_pred_total, model, full_model 

#%% EXTREME CLIMATE INDICES
 
def units_conversion(DS_exclim):
    da_list = []
    for feature in list(DS_exclim.keys()):
        if (type(DS_exclim[feature].values[0,0,0]) == np.timedelta64):
            print('Time')
            DS = timedelta_to_int(DS_exclim, feature)
        else:
            print('Integer')
            DS = DS_exclim[feature]
        
        da_list.append(DS)
    return xr.merge(da_list)    

# Start
start_date, end_date = '01-01-1976','31-12-2016'

DS_exclim_us = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_historical_us/monthly_data/*.nc').sel(time=slice(start_date, end_date))
DS_exclim_arg = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_historical_arg/monthly_data/*.nc').sel(time=slice(start_date, end_date))
DS_exclim_br = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_gswp3/monthly_data/*.nc').sel(time=slice(start_date, end_date))

#SHIFT 12 months forward
DS_exclim_br = DS_exclim_br.shift(time = 12)
DS_exclim_arg = DS_exclim_arg.shift(time = 12)

# COMBINE
DS_exclim_am = DS_exclim_us.combine_first(DS_exclim_arg)
DS_exclim_am = DS_exclim_am.combine_first(DS_exclim_br)
DS_exclim_am = DS_exclim_am.where(DS_y_obs_am_det.mean('time') > -10)
DS_exclim_am = rearrange_latlot(DS_exclim_am)

plot_2d_am_map(DS_exclim_am['txm'].mean('time'))

# New dataset
DS_exclim_am = DS_exclim_am.drop_vars(['fd','id','time_bnds','spi','spei','dtr','tr','tnlt2', 'tnltm2','tnltm20', 'tmlt5', 'tmge5', 'tmge10']) # Always zero
list_features = ['prcptot', 'txm'] # Most important variables: TR, TXGE35, TNM, TXM and precptot for water
DS_exclim_am = DS_exclim_am[list_features] 
# DS_exclim_am = DS_exclim_am.rename({'tr':'trop'}) #Only if TR is used to avoid confusion with dtr
df_list_features = list(DS_exclim_am.keys())

# Convert data from time to values (number of days)
DS_exclim_am_comb = units_conversion(DS_exclim_am)

# Adjust data
DS_exclim_am_comb = rearrange_latlot(DS_exclim_am_comb)
DS_exclim_am_comb = DS_exclim_am_comb.reindex(lat=DS_exclim_am_comb.lat[::-1])
if len(DS_exclim_am_comb.coords) >3 :
    DS_exclim_am_comb=DS_exclim_am_comb.drop('spatial_ref')
    
DS_exclim_am_det = DS_exclim_am_comb #detrend_dataset(DS_exclim_am_comb)

# =============================================================================
# Relative dates functions - shift and reshape
# =============================================================================
# Reshape to have each calendar year on the columns (1..12)
def reshape_data(dataframe):  #converts and reshape data
    #If already dataframe, skip the convertsion
    if isinstance(dataframe, pd.Series):    
        dataframe = dataframe.to_frame()
        
    dataframe['month'] = dataframe.index.get_level_values('time').month
    dataframe['year'] = dataframe.index.get_level_values('time').year
    dataframe.set_index('month', append=True, inplace=True)
    dataframe.set_index('year', append=True, inplace=True)
    # dataframe = dataframe.reorder_levels(['time', 'year','month'])
    dataframe.index = dataframe.index.droplevel('time')
    dataframe = dataframe.unstack('month')
    dataframe.columns = dataframe.columns.droplevel()
    return dataframe

def reshape_shift(dataset, shift_time=0):
    ### Convert to dataframe and shift according to input -> if shift time is 0, then nothing is shifted
    dataframe_1 = dataset.shift(time=-shift_time).to_dataframe()    
    
    # Define the column names based on the type of data - dataframe or dataarray
    if type(dataset) == xr.core.dataset.Dataset:
        print('dataset mode')
        column_names = [var_name +"_"+str(j) for var_name in dataset.data_vars 
                        for j in range(1 + shift_time, 13 + shift_time)]
        
    elif type(dataset) == xr.core.dataarray.DataArray: 
        print('dataArray mode') 
        column_names = [dataset.name +"_"+str(j) for j in range(1+shift_time,13+shift_time)]
    else:
        raise ValueError('Data must be either Dataset ot DataArray.')
        
    # Reshape dataframe
    dataframe_reshape = reshape_data(dataframe_1)
    dataframe_reshape.columns = column_names      
    return dataframe_reshape

def calendar_multiyear_adjust(month_list, df_entry, mode = "shift_one_year"):
    df = df_entry.copy()
    month_list = np.sort(month_list)[::-1]
    for month in month_list:
        if df.loc[df == month].sum() > 0:
            print(f'There are planting dates on the following year (y+1) for month {month}')
            if mode == "shift_one_year":
                df.loc[df == month] = 12 + month
            elif mode == 'erase':
                df.loc[df == month] = np.nan
        else:
            print(f'No planting dates for month {month}')
    return df.to_frame().dropna()


# =============================================================================
# Load calendars ############################################################
# =============================================================================
DS_cal_ggcmi = xr.open_dataset('../../paper_hybrid_agri/data/soy_rf_ggcmi_crop_calendar_phase3_v1.01.nc4') / (365/12)
DS_cal_sachs = xr.open_dataset('../../paper_hybrid_agri/data/Soybeans.crop.calendar_sachs_05x05.nc') / (365/12) 
DS_cal_mirca = xr.open_dataset('../../paper_hybrid_agri/data/mirca2000_soy_calendar.nc') # 

# =============================================================================
# TEST CALENDARS
# =============================================================================
DS_cal_mirca_subset_us = DS_cal_mirca.where(DS_y_obs_us_all['Yield'].mean('time') >= -10)
DS_cal_ggcmi_subset_us_test = DS_cal_ggcmi.where(DS_y_obs_us_all['Yield'].mean('time') >= -10)
DS_cal_ggcmi_subset_arg = DS_cal_ggcmi.where(DS_y_obs_arg['Yield'].mean('time') >= -10)
DS_cal_mirca_subset_arg_test = DS_cal_mirca.where(DS_y_obs_arg['Yield'].mean('time') >= -10)
DS_cal_ggcmi_subset_br = DS_cal_ggcmi.where(DS_y_obs_br['Yield'].mean('time') >= -10)
DS_cal_mirca_subset_br_test = DS_cal_mirca.where(DS_y_obs_br['Yield'].mean('time') >= -10)

plot_2d_am_map(DS_cal_mirca_subset_us['start'], title = 'MIRCA start')
plot_2d_am_map(DS_cal_ggcmi_subset_br['planting_day'], title = 'MIRCA start')
plot_2d_am_map(DS_cal_ggcmi_subset_arg['planting_day'], title = 'MIRCA start')

DS_calendar_combined = DS_cal_mirca_subset_us['start'].combine_first(DS_cal_ggcmi_subset_br['planting_day'])
DS_calendar_combined = DS_calendar_combined.combine_first(DS_cal_ggcmi_subset_arg['planting_day'])
DS_calendar_combined = rearrange_latlot(DS_calendar_combined)
plot_2d_am_map(DS_calendar_combined, title = 'combined start')

DS_cal_mirca_subset = DS_cal_mirca.where(DS_y_obs_am_det.mean('time') >= -10 )
DS_cal_sachs_month_subset = DS_cal_sachs.where(DS_y_obs_am_det.mean('time') >= -10)
DS_cal_ggcmi_subset = DS_cal_ggcmi.where(DS_y_obs_am_det.mean('time') >= -10)

# plot_2d_am_map(DS_cal_sachs_month_subset['plant'])
plot_2d_am_map(DS_cal_mirca_subset['start'], title = 'MIRCA start')
# plot_2d_am_map(DS_cal_mirca_subset['end'], title = 'MIRCA end')
# plot_2d_am_map(DS_cal_mirca_subset['end']-DS_cal_mirca_subset['start'] , title = 'MIRCA length')

plot_2d_am_map(DS_cal_ggcmi_subset['planting_day'], title = 'GGCMI start')
plot_2d_am_map(DS_cal_ggcmi_subset['maturity_day'], title = 'GGCMI maturity')
plot_2d_am_map(DS_cal_ggcmi_subset['growing_season_length'].dt.days, title = 'GGCMI length')
plot_2d_am_map(DS_cal_ggcmi_subset['data_source_used'], title = 'GGCMI length')

### Chose calendar:
DS_chosen_calendar_am = DS_cal_mirca_subset['start'].round() #DS_cal_ggcmi_subset['planting_day'].round() #DS_cal_mirca_subset['start'] #DS_calendar_combined # [ DS_cal_ggcmi_subset['planting_day']  #DS_cal_sachs_month_subset['plant'] DS_cal_mirca_subset['start'] ]
if DS_chosen_calendar_am.name != 'plant':
    DS_chosen_calendar_am = DS_chosen_calendar_am.rename('plant')

# # ATTENTION, REMOVING GRID CELLS THAT LOOK WEIRD AND IRRELEVANT. Specific to GGCMI
DS_chosen_calendar_am = DS_chosen_calendar_am.where(DS_chosen_calendar_am != 1, drop = True)

# Convert DS to df
df_chosen_calendar = DS_chosen_calendar_am.to_dataframe().dropna()
# Rounding up planting dates to closest integer in the month scale
df_calendar_month_am = df_chosen_calendar[['plant']].apply(np.rint).astype('Int64')

# transform the months that are early in the year to the next year (ex 1 -> 13). Attention as this should be done only for the south america region
df_calendar_month_am = calendar_multiyear_adjust([1,2], df_calendar_month_am['plant'])    

### LOAD climate date and clip to the calendar cells    
DS_exclim_am_det_clip = DS_exclim_am_det.sel(time=slice('1975-01-01','2016-12-16')).where(DS_chosen_calendar_am >= 0 )
plot_2d_am_map(DS_exclim_am_det_clip['prcptot'].mean('time'))
DS_exclim_am_det_clip.resample(time="1MS").mean(dim="time")

# =============================================================================
# CONVERT CLIMATIC VARIABLES ACCORDING TO THE SOYBEAN GROWING SEASON PER GRIDCELL 
# =============================================================================
# First reshape each year to make a 24 month calendar
df_clim_shift = reshape_shift(DS_exclim_am_det_clip)
df_clim_shift_12 = reshape_shift(DS_exclim_am_det_clip, shift_time = 12)
# Combine both dataframes and constraint it to be below 2016 just in case
df_test_reshape_twoyears = df_clim_shift.dropna().join(df_clim_shift_12).query('year <= 2016')
### Join and change name to S for the shift values
df_feature_reshape_shift = df_test_reshape_twoyears.dropna().join(df_calendar_month_am)
            
# Divide the dataset by climatic feature so the shifting does not mix the different variables together
list_df_feature_reshape_shift = []
for feature in list(DS_exclim_am_det_clip.keys()):
    df_feature_reshape_shift_var = pd.concat([df_feature_reshape_shift.loc[:,'plant'],
                                              df_feature_reshape_shift.filter(like=feature)], axis = 1)
    
    # Shift accoording to month indicator (hence +1) - SLOW
    df_feature_reshape_shift_var = (df_feature_reshape_shift_var.apply(lambda x : x.shift(-(int(x['plant']))+1) , axis=1)
                                .drop(columns=['plant']))
    
    list_df_feature_reshape_shift.append(df_feature_reshape_shift_var)
# Convert to dataframe
df_features_reshape_2years_am = pd.concat(list_df_feature_reshape_shift, axis=1)

### Select specific months ###################################################
suffixes = tuple(["_"+str(j) for j in range(3,6)])
df_feature_season_6mon_am = df_features_reshape_2years_am.loc[:,df_features_reshape_2years_am.columns.str.endswith(suffixes)]

# Organising the data
df_feature_season_6mon_am = df_feature_season_6mon_am.rename_axis(index={'year':'time'}).reorder_levels(['time','lat','lon']).sort_index()
df_feature_season_6mon_am = df_feature_season_6mon_am.where(df_obs_am_det['Yield']>=0).dropna().astype(float)

# SECOND DETRENDING PART - SEASONAL
DS_feature_season_6mon_am = xr.Dataset.from_dataframe(df_feature_season_6mon_am)
DS_feature_season_6mon_am_det = detrend_dataset(DS_feature_season_6mon_am, deg = 'free')
df_feature_season_6mon_am_det = DS_feature_season_6mon_am_det.to_dataframe().dropna()

for feature in df_feature_season_6mon_am_det.columns:
    df_feature_season_6mon_am[feature].groupby('time').mean().plot(label = 'old')
    df_feature_season_6mon_am_det[feature].groupby('time').mean().plot(label = 'detrend')
    # df_feature_season_6mon_us_nodet[feature].groupby('time').mean().plot(label = '1 detrend')
    plt.title(f'{feature}')
    plt.legend()
    plt.show()
    print(np.round(df_feature_season_6mon_am_det[feature].groupby('time').mean().max(),3))

# =============================================================================
# # Detrending in season
# =============================================================================
df_feature_season_6mon_am = df_feature_season_6mon_am_det

list_feat_precipitation = [s for s in df_feature_season_6mon_am.keys() if "prcptot" in s]
for feature in list_feat_precipitation:
    df_feature_season_6mon_am[feature][df_feature_season_6mon_am[feature] < 0] = 0

# Remove any possible missing value but mostly the 2016 year
df_obs_am_det_clip = df_obs_am_det.where(df_feature_season_6mon_am['prcptot'+suffixes[0]] > -100).dropna().reorder_levels(['time','lat','lon']).sort_index()

feature_importance_selection(df_feature_season_6mon_am, df_obs_am_det_clip['Yield']) 
 
print('Dynamic ECE results:') 
X, y = df_feature_season_6mon_am, df_obs_am_det_clip['Yield']
y_pred_exclim_dyn_am, y_pred_total_exclim_dyn_am, model_exclim_dyn_am, full_model_exclim_dyn_am = calibration(X, y)

# =============================================================================
# # Turn on if the selection of variables in of interest
# =============================================================================
# df_feature_performance = pd.DataFrame(columns=['performance'], index = df_list_features)
# for feature in df_list_features[0:]: #Remove other pcptot influence
#     print('feature is', feature)
#     X = df_feature_season_6mon_am.loc[:,[feature+'_3', feature + '_4', feature + '_5' ] ] 
    
#     y_pred_exclim_dyn_am, y_pred_total_exclim_dyn_am, model_exclim_dyn_am, full_model_exclim_dyn_am = calibration(X, y)
    
#     X_config, y_config = shuffle(X, y, random_state=0)
#     X_train, X_test, y_train, y_test = train_test_split(X_config, y_config, test_size=0.1, random_state=0)
    
#     # Test performance
#     y_pred = model_exclim_dyn_am.predict(X_test)
    
#     # report performance
#     df_feature_performance.loc[feature] = round(r2_score(y_test, y_pred),2)


#%% EPIC RF
df_epic_am_det_clip = df_epic_am_det.where(df_feature_season_6mon_am['prcptot'+suffixes[0]] > -100).dropna().reorder_levels(['time','lat','lon']).sort_index()

X, y = df_epic_am_det_clip, df_obs_am_det_clip['Yield']

# Standard model
print('Standard Epic results:')
y_pred_epic_am, y_pred_total_epic_am, model_epic_am, full_model_epic_am  = calibration(X, y)

#%% Hybrid model
# Combine the EPIC output with the Extreme climate indices to generate the input dataset for the hybrid model
df_input_hybrid_am = pd.concat([df_epic_am_det_clip, df_feature_season_6mon_am], axis = 1)

# Save this for future operations:
df_input_hybrid_am.to_csv('dataset_input_hybrid_am_forML.csv')
df_obs_am_det_clip.to_csv('dataset_obs_yield_am_forML.csv')

# Feature selection
feature_importance_selection(df_input_hybrid_am, df_obs_am_det_clip)

# Evaluate Model
print('Standard Hybrid results:')
X, y = df_input_hybrid_am, df_obs_am_det_clip['Yield']

y_pred_hyb_am_rf, y_pred_total_hyb_am_rf, model_hyb_am_rf, full_model_hyb2_am_rf = calibration(X, y, type_of_model='RF')
# Deep neural network - takes some time to run 
y_pred_hyb_am2, y_pred_total_hyb_am2, model_hyb_am2, full_model_hyb_am2 = calibration(X, y, type_of_model='DNN')

# plot_model(full_model_hyb_am2, to_file='paper_figures/model_plot.png', show_shapes=True, show_layer_names=True)

# # =============================================================================
# # Save the Model 
# # =============================================================================
# import pickle

# model_hyb_am2['estimator'].model_.save('hybrid_notfull_ANN_3x400_AMER_r71.pkl')
# full_model_hyb_am2['estimator'].model_.save('hybrid_ANN_3x400_AMER_r71.pkl')

# # Model 2 ---------------------------------------------------------------------
# # Save the Keras model first:
# model_hyb_am2['estimator'].model_.save('keras_notfull_model_AMER_r71.h5')
# full_model_hyb_am2['estimator'].model_.save('keras_model_AMER_r71.h5')

# # This hack allows us to save the sklearn pipeline:
# model_hyb_am2['estimator'].model = None
# full_model_hyb_am2['estimator'].model = None

# import joblib
# from keras.models import load_model
# # Finally, save the pipeline:
# joblib.dump(model_hyb_am2, 'sklearn_pipeline_notfull_AMER_r71.pkl')
# joblib.dump(full_model_hyb_am2, 'sklearn_pipeline_AMER_r71.pkl')


# # #####
# model_hyb_am2_test = joblib.load('sklearn_pipeline_notfull_AMER_r2068_2.pkl')    
# model_hyb_am2_test['estimator'].model_ = load_model('best_model_test.h5')

# full_model_hyb_am2_test = joblib.load('sklearn_pipeline_AMER_r2068_2.pkl')    
# full_model_hyb_am2_test['estimator'].model_ = load_model('best_model_full.h5')

# test_perf(X,y, model_hyb_am2) # Traditional training
# test_perf(X,y, model_hyb_test) # Traditional training
# test_perf(X,y, model_hyb_am2_test) # Traditional training

# test_perf(X,y, full_model_hyb_am2) # Full Traditional training
# test_perf(X,y, full_model_hyb_test) # Full Traditional training
# test_perf(X,y, full_model_hyb_am2_test) # Full Traditional training

# from keras.models import load_model
# full_model_hyb = joblib.load('sklearn_pipeline_AMER_r2067_2.pkl')
# # model_hyb_am2 = joblib.load('sklearn_pipeline_notfull_AMER_r2068.pkl')
# # Then, load the Keras model:
# full_model_hyb['estimator'].model = load_model('keras_model_AMER_r2067_2.h5')
# # model_hyb_am2['estimator'].model = load_model('keras_notfull_model_AMER_r2068.h5')

#%% Validation and performance of the models

def conversion_dataframe_dataset(df_input, model, df_base = df_obs_am_det_clip):
    df_1 = df_base.copy()
    predictions = model.predict(df_input.values)
    df_1.loc[:,'Yield'] = predictions 
    DS = xr.Dataset.from_dataframe(df_1)
    DS = rearrange_latlot(DS)
    return df_1, DS

# Convert the dataframes into Datasets
df_predict_hyb_am, DS_predict_test_hist = conversion_dataframe_dataset(df_input_hybrid_am, full_model_hyb_am2, df_base = df_obs_am_det_clip)
df_predict_epic_hist, DS_predict_epic_hist = conversion_dataframe_dataset(df_epic_am_det_clip, full_model_epic_am, df_base = df_obs_am_det_clip)
df_predict_clim_hist, DS_predict_clim_hist = conversion_dataframe_dataset(df_feature_season_6mon_am, full_model_exclim_dyn_am, df_base = df_obs_am_det_clip)
    
# =============================================================================
# # save .nc files for AM, US, BR, ARG for the hybrid predictions
# =============================================================================
DS_predict_test_hist.to_netcdf('output_models_am/hybrid_epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc')
DS_predict_test_hist.where(DS_y_obs_us_all['Yield'] > -1).to_netcdf('output_models_am/hybrid_epic_us-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc')
DS_predict_test_hist.where(DS_y_obs_br['Yield'] > -1).to_netcdf('output_models_am/hybrid_epic_br-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc')
DS_predict_test_hist.where(DS_y_obs_arg['Yield'] > -1).to_netcdf('output_models_am/hybrid_epic_arg-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc')

shift_2012 = DS_predict_test_hist['Yield'].sel(time=2012) / DS_predict_test_hist['Yield'].mean(['time']) 


# =============================================================================
# Regularisation of the data
# =============================================================================
DS_y_obs_am_det_regul = DS_y_obs_am_det.where(DS_predict_test_hist['Yield']> -10)
DS_y_epic_am_det_regul = DS_y_epic_am_det.where(DS_predict_test_hist['Yield']> -10)
plot_2d_am_map(DS_y_obs_am_det_regul.mean(['time']), title = 'Observed')
plot_2d_am_map(DS_predict_test_hist['Yield'].mean(['time']), title = 'Hybrid') # Should be the same area

# Difference between the event anomaly and the predicted by the hybrid model
plot_2d_am_map(DS_predict_test_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012), colormap = 'RdBu', vmin=-1, vmax = 1)
plot_2d_am_map(DS_predict_test_hist['Yield'].sel(time=2012) - DS_predict_test_hist['Yield'].mean('time'), colormap = 'RdBu', title = '2012 event anomaly')

# PLOTS
plt.figure(figsize=(10,6), dpi=300) #plot clusters
plt.plot(DS_y_epic_am_det_regul.time, DS_y_epic_am_det_regul.mean(['lat','lon']), label = 'Original EPIC',linestyle='dashed',linewidth=3)
plt.plot(DS_y_obs_am_det_regul.time, DS_y_obs_am_det_regul.mean(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_predict_epic_hist['Yield'].mean(['lat','lon']), label = 'RF:EPIC')
plt.plot(DS_predict_clim_hist.time, DS_predict_clim_hist['Yield'].mean(['lat','lon']), label = 'RF:ECE')
plt.plot(DS_predict_test_hist.time, DS_predict_test_hist['Yield'].mean(['lat','lon']), label = 'RF:Hybrid')
plt.title('Absolute shocks')
plt.legend()
plt.show()

# Scaled
plt.figure(figsize=(10,6), dpi=300) #plot clusters
plt.plot(DS_y_epic_am_det_regul.time, DS_y_epic_am_det_regul.mean(['lat','lon'])/DS_y_epic_am_det_regul.mean(), label = 'Original EPIC',linestyle='dashed',linewidth=3)
plt.plot(DS_y_obs_am_det_regul.time, DS_y_obs_am_det_regul.mean(['lat','lon'])/DS_y_obs_am_det_regul.mean(), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_predict_epic_hist['Yield'].mean(['lat','lon'])/DS_predict_epic_hist['Yield'].mean(), label = 'RF:EPIC')
plt.plot(DS_predict_clim_hist.time, DS_predict_clim_hist['Yield'].mean(['lat','lon'])/DS_predict_clim_hist['Yield'].mean(), label = 'RF:ECE')
plt.plot(DS_predict_test_hist.time, DS_predict_test_hist['Yield'].mean(['lat','lon'])/DS_predict_test_hist['Yield'].mean(), label = 'RF:Hybrid')
plt.title('Scaled shocks')
plt.legend()
plt.show()

# =============================================================================
# ### WIEGHTED ANALYSIS
# =============================================================================

# Weighted comparison for each model - degree of explanation
DS_y_epic_am_weighted = weighted_conversion(DS_y_epic_am_det_regul, DS_mirca_test)
DS_predict_epic_weighted = weighted_conversion(DS_predict_epic_hist['Yield'], DS_mirca_test)
DS_predict_clim_am_weighted = weighted_conversion(DS_predict_clim_hist['Yield'], DS_mirca_test)
DS_predict_hyb_am_weighted = weighted_conversion(DS_predict_test_hist['Yield'], DS_mirca_test)

# Weighted plot
plt.figure(figsize=(8,5), dpi=300) #plot clusters
# plt.plot(DS_predict_epic_hist.time, DS_epic_orig_weighted['Yield'].sum(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_y_obs_det_weighted_am.time, DS_y_obs_det_weighted_am['Yield'], label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_weighted.time, DS_predict_epic_weighted['Yield'], label = 'RF:EPIC')
plt.plot(DS_predict_clim_am_weighted.time, DS_predict_clim_am_weighted['Yield'], label = 'RF:CLIM')
plt.plot(DS_predict_hyb_am_weighted.time, DS_predict_hyb_am_weighted['Yield'], label = 'RF:hybrid')
plt.ylabel('Yield (ton/ha)')
plt.xlabel('Years')
plt.title('Weighted analysis')
plt.legend()
plt.show()

# Plot scatter plots of the errors
for df_version, label_to_be_used in zip([df_epic_am_det['yield'], df_predict_epic_hist['Yield'], df_predict_clim_hist['Yield'], df_predict_hyb_am['Yield']],
                                        ['Original EPIC','RF:EPIC', 'RF:CLIM','Hybrid' ]):
    if len(df_version) == len(df_obs_am_det):
        ds_ref = df_obs_am_det
    elif len(df_version) == len(df_obs_am_det_clip):
        ds_ref = df_obs_am_det_clip

    # Scatter plots
    plt.figure(figsize=(5,5), dpi=250) #plot clusters
    plt.scatter(ds_ref['Yield'], df_version)
    plt.plot(ds_ref['Yield'].sort_values(), ds_ref['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
    plt.ylabel(f'{label_to_be_used} predicted yield')
    plt.xlabel('Observed yield')
    plt.legend()
    # plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
    plt.show()
   
X_train, X_test, y_train, y_test = train_test_split(df_input_hybrid_am, df_obs_am_det_clip['Yield'], test_size=0.1, random_state=0)

X_analysis, y_analysis = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_analysis, y_analysis, test_size=0.1, random_state=0)

df_epic_am_det_clip = df_epic_am_det.where(df_obs_am_det_clip['Yield'] > -1).dropna()
X_epic_analysis = shuffle(df_epic_am_det_clip, random_state=0)
X_epic_train, X_epic_test = train_test_split(X_epic_analysis, test_size=0.1, random_state=0)


# Gridded comparison for each model during test set (OoS)  - degree of explanation - if error shows up, check test_size
print("R2 OBS-RF:EPIC test set:",round(r2_score(y_test, y_pred_epic_am),2))
print("R2 OBS-RF:Clim test set:",round(r2_score(y_test, y_pred_exclim_dyn_am),2))
print("R2 OBS-Hybrid test set:",round(r2_score(y_test, y_pred_hyb_am2),2))
print("_______________________________________")

results_performance_oot = pd.DataFrame([[round(r2_score(y_test, X_epic_test),2), round(mean_absolute_error(y_test, X_epic_test),3), round(mean_squared_error(y_test, X_epic_test, squared=False),3)],
                                        [round(r2_score(y_test, y_pred_epic_am),2), round(mean_absolute_error(y_test, y_pred_epic_am),3), round(mean_squared_error(y_test, y_pred_epic_am, squared=False),3)],
                                       [round(r2_score(y_test, y_pred_exclim_dyn_am),2), round(mean_absolute_error(y_test, y_pred_exclim_dyn_am),3), round(mean_squared_error(y_test, y_pred_exclim_dyn_am, squared=False),3)],
                                       [round(r2_score(y_test, y_pred_hyb_am2),2), round(mean_absolute_error(y_test, y_pred_hyb_am2),3), round(mean_squared_error(y_test, y_pred_hyb_am2, squared=False),3)]],
                                       columns = ['R2','MAE','RMSE'], index = ['EPIC model', 'Stat. model EPIC','Stat. model climate indices', 'Hybrid model'])


# Gridded comparison for each model with full model (NOT OoS)  - degree of explanation - if error shows up, check test_size
print("R2 OBS-EPIC_original on entire set:",round(r2_score(df_obs_am_det['Yield'], df_epic_am_det['yield']),2))
print("R2 OBS-RF:EPIC full model on entire set:",round(r2_score(y, y_pred_total_epic_am),2))
print("R2 OBS-RF:Clim full model on entire set:",round(r2_score(y, y_pred_total_exclim_dyn_am),2))
print("R2 OBS-Hybrid full model on entire set:",round(r2_score(y, y_pred_total_hyb_am2),2))
print("_______________________________________")

print("Weighted R2 OBS-EPIC:",round(r2_score(DS_y_obs_det_weighted_am['Yield'].sel(time = slice(1975,2015)), DS_y_epic_am_weighted['Yield'].sel(time = slice(1975,2015))),2))
print("Weighted R2 OBS-RF:EPIC:",round(r2_score(DS_y_obs_det_weighted_am['Yield'].sel(time = slice(1975,2015)), DS_predict_epic_weighted['Yield'].sel(time = slice(1975,2015))),2))
print("Weighted R2 OBS-Clim:",round(r2_score(DS_y_obs_det_weighted_am['Yield'].sel(time = slice(1975,2015)), DS_predict_clim_am_weighted['Yield'].sel(time = slice(1975,2015))),2))
print("Weighted R2 OBS-Hybrid:",round(r2_score(DS_y_obs_det_weighted_am['Yield'].sel(time = slice(1975,2015)), DS_predict_hyb_am_weighted['Yield'].sel(time = slice(1975,2015))),2))
print("_______________________________________")

# =============================================================================
# # Extreme case 2012 error
# =============================================================================
plot_2d_am_map(DS_y_epic_am_det_regul.sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012), colormap = 'RdBu', vmin = -1, vmax = 1, title = "a) EPIC-IIASA")
plot_2d_am_map(DS_predict_epic_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012), colormap = 'RdBu', vmin = -1, vmax = 1, title = "b) Stat-EPIC")
plot_2d_am_map(DS_predict_clim_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012), colormap = 'RdBu', vmin = -1, vmax = 1, title = "c) Stat-clim")

DS_predict_test_hist_plot = DS_predict_test_hist.copy()
DS_predict_test_hist_plot = DS_predict_test_hist_plot.rename({'Yield':'Yield (ton/ha)'})

DS_anomaly_bias = ( DS_predict_test_hist_plot['Yield (ton/ha)'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012) ) #/ DS_y_obs_am_det_regul.sel(time=2012)
DS_anomaly_2012 = ( DS_predict_test_hist_plot['Yield (ton/ha)'].sel(time=2012) - DS_predict_test_hist_plot['Yield (ton/ha)'].sel(time = slice(2000,2015)).mean('time') ) #/ DS_predict_test_hist_plot['Yield (ton/ha)'].sel(time = slice(2000,2015)).mean('time') 
DS_anomaly_2012_globiom = ( DS_predict_test_hist['Yield'].sel(time=2012) / DS_predict_test_hist['Yield'].sel(time = slice(2000,2015)).mean('time') ) #/ DS_predict_test_hist_plot['Yield (ton/ha)'].sel(time = slice(2000,2015)).mean('time') 
DS_anomaly_2012_globiom.to_netcdf('output_models_am/shifters_2012_hybrid_climatology.nc')

plot_2d_am_map(DS_anomaly_bias, colormap = 'RdBu', vmin = -1, vmax = 1, title = "a) 2012 bias")
plot_2d_am_map(DS_anomaly_2012, colormap = 'RdBu', vmin = -1, vmax = 1, title = "b) Simulation of 2012 anomaly")


print('Sum of squared errors for EPIC case',(((DS_predict_epic_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values)
print('Sum of squared errors for CLIM case',(((DS_predict_clim_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values)
print('Sum of squared errors for Hybrid case',(((DS_predict_test_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values)

print('The Hybrid model shows',round(( ( (((DS_predict_test_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values -
                                        (((DS_predict_epic_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values ) / 
      (((DS_predict_epic_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values) * 100,2) , '% reduction with respect to the EPIC model')

print('The Hybrid model shows',round(( ((((DS_predict_test_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values -
                                       (((DS_predict_clim_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values ) / 
      (((DS_predict_clim_hist['Yield'].sel(time=2012) - DS_y_obs_am_det_regul.sel(time=2012))**2).sum()).values) * 100,2), '% reduction with respect to the CLIM model' )


# =============================================================================
# Comparison per country
# =============================================================================
# Plot historical timeline of weighted soybean yield
plt.plot(DS_y_obs_det_weighted_us.sel(time = slice(1975,2015)).time, DS_y_obs_det_weighted_us['Yield'].sel(time = slice(1975,2015)), label = 'US')
plt.plot(DS_y_obs_det_weighted_br.sel(time = slice(1975,2015)).time, DS_y_obs_det_weighted_br['Yield'].sel(time = slice(1975,2015)), label = 'BR')
plt.plot(DS_y_obs_det_weighted_arg.sel(time = slice(1975,2015)).time, DS_y_obs_det_weighted_arg['Yield'].sel(time = slice(1975,2015)), label = 'ARG')
plt.legend()
plt.ylim(1,3)
plt.title('Observed data')
plt.show()

# Plot each country detrended and weighted
DS_y_hybrid_weighted_us = weighted_conversion(DS_predict_test_hist['Yield'], DS_area = DS_mirca_test.where(DS_y_obs_us_all['Yield'] > -1))
DS_y_hybrid_weighted_br = weighted_conversion(DS_predict_test_hist['Yield'], DS_area = DS_mirca_test.where(DS_y_obs_br['Yield'] > -1))
DS_y_hybrid_weighted_arg = weighted_conversion(DS_predict_test_hist['Yield'], DS_area = DS_mirca_test.where(DS_y_obs_arg['Yield'] > -1))

# Plot historical timeline of weighted soybean yield
plt.plot(DS_y_hybrid_weighted_us.time, DS_y_hybrid_weighted_us['Yield'], label = 'US')
plt.plot(DS_y_hybrid_weighted_br.time, DS_y_hybrid_weighted_br['Yield'], label = 'BR')
plt.plot(DS_y_hybrid_weighted_arg.time, DS_y_hybrid_weighted_arg['Yield'], label = 'ARG')
plt.axhline(y = DS_y_obs_det_weighted_us['Yield'].sel(time = 2012), linestyle = 'dashed')
plt.axhline(y = DS_y_obs_det_weighted_br['Yield'].sel(time = 2012), linestyle = 'dashed')
plt.axhline(y = DS_y_obs_det_weighted_arg['Yield'].sel(time = 2012), linestyle = 'dashed')
plt.title('Hybrid data')
plt.ylim(1,3)
plt.legend()
plt.show()

print("Weighted R2 US:",round(r2_score(DS_y_obs_det_weighted_us['Yield'].sel(time = slice(1975,2015)),DS_y_hybrid_weighted_us['Yield']), 2))
print("Weighted R2 BR:",round(r2_score(DS_y_obs_det_weighted_br['Yield'].sel(time = slice(1975,2015)),DS_y_hybrid_weighted_br['Yield']), 2))
print("Weighted R2 ARG:",round(r2_score(DS_y_obs_det_weighted_arg['Yield'].sel(time = slice(1975,2015)),DS_y_hybrid_weighted_arg['Yield']), 2))

# Absolute shocks
plt.bar(x = 'US', height = DS_y_obs_det_weighted_us['Yield'].sel(time = 2012) - DS_y_obs_det_weighted_us['Yield'].sel(time = slice(1975,2015)).mean('time').values)
plt.bar(x = 'BR', height = DS_y_obs_det_weighted_br['Yield'].sel(time = 2012) - DS_y_obs_det_weighted_br['Yield'].sel(time = slice(1975,2015)).mean('time').values)
plt.bar(x = 'ARG', height = DS_y_obs_det_weighted_arg['Yield'].sel(time = 2012) - DS_y_obs_det_weighted_arg['Yield'].sel(time = slice(1975,2015)).mean('time').values)
plt.axhline(y = DS_y_obs_det_weighted_am['Yield'].sel(time = 2012).values - DS_y_obs_det_weighted_am['Yield'].sel(time = slice(1975,2015)).mean('time').values, color = 'black', linestyle = 'dashed', label = 'Aggregated')
plt.title('2012 absolute shock')
plt.ylabel('Yield anomaly (ton/ha)')
plt.legend()
plt.show()


# Plot production - TEST
def production(DS, DS_area):
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'] ) ).to_dataset(name = 'Yield')
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = ((DS * DS_area['harvest_area'] ) )
    return DS_weighted.sum(['lat','lon'])

DS_produ_am = production(DS_y_obs_am_det_regul, DS_mirca_test.where(DS_y_obs_am_det_regul > -1))
DS_produ_us = production(DS_y_obs_am_det_regul, DS_mirca_test.where(DS_y_obs_us_all['Yield'] > -1))
DS_produ_br = production(DS_y_obs_am_det_regul, DS_mirca_test.where(DS_y_obs_br['Yield'] > -1))
DS_produ_arg = production(DS_y_obs_am_det_regul, DS_mirca_test.where(DS_y_obs_arg['Yield'] > -1))


plt.plot(DS_produ_us.time, DS_produ_us['Yield'], label = 'US')
plt.plot(DS_produ_br.time, DS_produ_br['Yield'], label = 'BR')
plt.plot(DS_produ_arg.time, DS_produ_arg['Yield'], label = 'ARG')
plt.legend()
plt.show()

plt.plot(DS_produ_am.time, DS_produ_am['Yield'], label = 'AM', color = 'black')
plt.stackplot(DS_produ_am.time, DS_produ_us['Yield'], DS_produ_br['Yield'], DS_produ_arg['Yield'], labels = ['US', 'BR', 'ARG'])
plt.ylabel('Production (tonnes)')
plt.legend()
plt.tight_layout()
plt.show()





