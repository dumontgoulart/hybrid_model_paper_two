# -*- coding: utf-8 -*-
"""
Load factual
Load counterfactuals
Statistical comparisons

Created on Tue Mar 29 10:15:06 2022

@author: morenodu
"""
import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

# LOAD FUNCTIONS
def weighted_conversion(DS, DS_area, name_ds = 'Yield'):
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) ) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])).to_dataset(name = name_ds)
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])))
    return DS_weighted.sum(['lat','lon'])

def rearrange_latlot(DS, resolution = 0.5):
    DS = DS.sortby('lat')
    DS = DS.sortby('lon')
    new_lat = np.arange(DS.lat.min(), DS.lat.max() + resolution, resolution)
    new_lon = np.arange(DS.lon.min(), DS.lon.max() + resolution, resolution)
    DS = DS.reindex({'lat':new_lat})
    DS = DS.reindex({'lon':new_lon})
    return DS

countries = shpreader.natural_earth(resolution='50m',category='cultural',name='admin_0_countries')
# Find the boundary polygon.
for country in shpreader.Reader(countries).records():
    if country.attributes['SU_A3'] == 'ARG':
        arg_border0 = country.geometry
    elif country.attributes['SU_A3'] == 'BRA':
        bra_border0 = country.geometry
    elif country.attributes['SU_A3'] == 'USA':
        usa_border0 = country.geometry
        
# Plots for spatial distribution of anomalies
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

# plot multiple figures in a row - counterfactuals
def plot_2d_am_multi(DS, map_proj = None, map_title = None):
    if map_proj is None:
        map_proj = ccrs.LambertAzimuthalEqualArea(central_longitude=-74.5, central_latitude=0)
    else:
        map_proj = map_proj
    
    if len(DS.time) > 1:
        p = DS.plot(transform=ccrs.PlateCarree(), col="time", col_wrap=5, subplot_kws={"projection": map_proj}, 
                    robust=True, cmap = 'RdBu', vmin=-1., vmax=1., zorder=20, aspect=0.9)
        
        for ax in p.axes.flat:
            ax.axis('off')
            ax.add_feature(cartopy.feature.LAND, facecolor='gray', alpha=0.1)
            # ax.add_feature(cartopy.feature.BORDERS, alpha = 0.1)
            ax.add_geometries([usa_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
            ax.add_geometries([bra_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
            ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
            # ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='None', alpha=0.5, lw=0.7, zorder=21)
            ax.set_extent([-113,-36,-38,45])
            ax.set_aspect('equal', 'box')
            
            if map_title is not None:
                plt.suptitle(map_title)        
        plt.show()
            
    elif len(DS.time) == 1:
        plot_2d_am_map(DS, title = map_title, colormap = 'RdBu', vmin = None, vmax = None)
        plt.show()
    
    elif len(DS.time) == 0:
        print(f'No counterfactuals for this scenario {DS.name}')
    
#%% Load historical case
DS_historical_hybrid = xr.load_dataset("output_models_am/hybrid_epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
DS_historical_hybrid = rearrange_latlot(DS_historical_hybrid)

### Load future hybrid runs - detrended:
DS_hybrid_gfdl_26 = xr.load_dataset("output_models_am/hybrid_gfdl-esm4_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_gfdl_85 = xr.load_dataset("output_models_am/hybrid_gfdl-esm4_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_ipsl_26 = xr.load_dataset("output_models_am/hybrid_ipsl-cm6a-lr_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_ipsl_85 = xr.load_dataset("output_models_am/hybrid_ipsl-cm6a-lr_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_ukesm_26 = xr.load_dataset("output_models_am/hybrid_ukesm1-0-ll_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_ukesm_85 = xr.load_dataset("output_models_am/hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_2015_2100.nc")
# Merge all scenarios
DS_hybrid_all = xr.merge([DS_hybrid_gfdl_26.rename({'yield-soy-noirr':'gfdl_26'}),DS_hybrid_gfdl_85.rename({'yield-soy-noirr':'gfdl_85'}),
                          DS_hybrid_ipsl_26.rename({'yield-soy-noirr':'ipsl_26'}),DS_hybrid_ipsl_85.rename({'yield-soy-noirr':'ipsl_85'}),
                          DS_hybrid_ukesm_26.rename({'yield-soy-noirr':'ukesm_26'}),DS_hybrid_ukesm_85.rename({'yield-soy-noirr':'ukesm_85'})])

#### Load future hybrid runs:
DS_hybrid_trend_gfdl_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_gfdl-esm4_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_gfdl_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_gfdl-esm4_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ipsl_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ipsl-cm6a-lr_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ipsl_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ipsl-cm6a-lr_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ukesm_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ukesm1-0-ll_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ukesm_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ukesm1-0-ll_ssp585_default_yield_soybean_2015_2100.nc")
# Merge all scenarios
DS_hybrid_trend_all = xr.merge([DS_hybrid_trend_gfdl_26.rename({'yield-soy-noirr':'gfdl_26'}),DS_hybrid_trend_gfdl_85.rename({'yield-soy-noirr':'gfdl_85'}),
                          DS_hybrid_trend_ipsl_26.rename({'yield-soy-noirr':'ipsl_26'}),DS_hybrid_trend_ipsl_85.rename({'yield-soy-noirr':'ipsl_85'}),
                          DS_hybrid_trend_ukesm_26.rename({'yield-soy-noirr':'ukesm_26'}),DS_hybrid_trend_ukesm_85.rename({'yield-soy-noirr':'ukesm_85'})])

# Mean and standard deviation of each projections at grid level:
print(f'Mean historical:{round(DS_historical_hybrid.to_dataframe().mean().values.item(),2)} and Std: {round(DS_historical_hybrid.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_gfdl_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_gfdl_85.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_ipsl_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_ipsl_85.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_ukesm_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_ukesm_85.to_dataframe().std().values.item(),2)}')

#%% WIEGHTED ANALYSIS
# =============================================================================

### Use MIRCA to isolate the rainfed 90% soybeans
DS_mirca = xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).rename({'latitude': 'lat', 'longitude': 'lon','annual_area_harvested_rfc_crop08_ha_30mn':'harvest_area'})

#### HARVEST DATA
DS_harvest_area_sim = xr.load_dataset("../../paper_hybrid_agri/data/soybean_harvest_area_calculated_americas_hg.nc", decode_times=False)
DS_harvest_area_sim = DS_harvest_area_sim.sel(time = slice(2014,2016)).mean('time') #.sel(time = slice(1979,2016))
DS_harvest_area_sim = DS_harvest_area_sim.where(DS_mirca['harvest_area'] > 0 )

# HIstorical, change with year
DS_harvest_area_hist = DS_harvest_area_sim.where(DS_historical_hybrid['Yield']> -5)
DS_harvest_area_hist = rearrange_latlot(DS_harvest_area_hist)

# Future, it works as the constant area throught the 21st century based on 2014/15/16
DS_harvest_area_fut = DS_harvest_area_sim #.sel(time=slice(2014,2016)).mean(['time'])
DS_harvest_area_fut = rearrange_latlot(DS_harvest_area_fut)

# Test plots to check for problems
plot_2d_am_map(DS_harvest_area_hist['harvest_area'].isel(time = 0))
plot_2d_am_map(DS_harvest_area_hist['harvest_area'].isel(time = -1))
plot_2d_am_map(DS_harvest_area_fut['harvest_area'], title = 'Future projections')

# Weighted comparison for each model - degree of explanation
DS_historical_hybrid_weighted = weighted_conversion(DS_historical_hybrid['Yield'], DS_area = DS_harvest_area_hist)

# Future projections and transform into weighted timeseries
DS_hybrid_gfdl_26_weighted = weighted_conversion(DS_hybrid_gfdl_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_gfdl_85_weighted = weighted_conversion(DS_hybrid_gfdl_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ipsl_26_weighted = weighted_conversion(DS_hybrid_ipsl_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ipsl_85_weighted = weighted_conversion(DS_hybrid_ipsl_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ukesm_26_weighted = weighted_conversion(DS_hybrid_ukesm_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ukesm_85_weighted = weighted_conversion(DS_hybrid_ukesm_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)

DS_historical_hybrid_weighted['Yield'].plot(label = 'history')
DS_hybrid_ukesm_85_weighted['Yield'].plot(label = 'ukesm 85')
plt.legend()
plt.show()

plt.figure(figsize=(8,6), dpi=300) #plot clusters
plt.axhline(y = DS_historical_hybrid_weighted['Yield'].min().values, linestyle = 'dashed', label = 'Factual')
DS_hybrid_gfdl_26_weighted['Yield'].plot(label = 'gfdl 26')
DS_hybrid_gfdl_85_weighted['Yield'].plot(label = 'gfdl 85')
DS_hybrid_ipsl_26_weighted['Yield'].plot(label = 'ipsl 26')
DS_hybrid_ipsl_85_weighted['Yield'].plot(label = 'ipsl 85')
DS_hybrid_ukesm_26_weighted['Yield'].plot(label = 'ukesm 26')
DS_hybrid_ukesm_85_weighted['Yield'].plot(label = 'ukesm 85')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5), dpi=300) #plot clusters
sns.kdeplot(DS_historical_hybrid_weighted['Yield'], label = 'History', fill = True)
sns.kdeplot(DS_hybrid_gfdl_26_weighted['Yield'], label = 'gfdl 26', fill = True)
sns.kdeplot(DS_hybrid_gfdl_85_weighted['Yield'], label = 'gfdl 85', fill = True)
sns.kdeplot(DS_hybrid_ipsl_26_weighted['Yield'], label = 'ipsl 26', fill = True)
sns.kdeplot(DS_hybrid_ipsl_85_weighted['Yield'], label = 'ipsl 85', fill = True)
sns.kdeplot(DS_hybrid_ukesm_26_weighted['Yield'], label = 'ukesm 26', fill = True)
sns.kdeplot(DS_hybrid_ukesm_85_weighted['Yield'], label = 'ukesm 85', fill = True)
plt.legend()
plt.show()

# put the scenarios all together
DS_hybrid_all_weighted = weighted_conversion(DS_hybrid_all, DS_area = DS_harvest_area_fut)
df_hybrid_weighted_melt = pd.melt(DS_hybrid_all_weighted.to_dataframe(),ignore_index= False )
df_hybrid_weighted_melt_counterfactuals = df_hybrid_weighted_melt.where(df_hybrid_weighted_melt['value'] <= DS_historical_hybrid_weighted['Yield'].min().values )

df_hybrid_weighted_melt_counterfactuals_split = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > - 10].copy()
df_hybrid_weighted_melt_counterfactuals_split['RCP'] = df_hybrid_weighted_melt_counterfactuals_split.variable.str.split('_').str[-1]


# put the scenarios all together
DS_hybrid_trend_all_weighted = weighted_conversion(DS_hybrid_trend_all, DS_area = DS_harvest_area_fut)
df_hybrid_trend_weighted_melt = pd.melt(DS_hybrid_trend_all_weighted.to_dataframe(),ignore_index= False )
df_hybrid_trend_weighted_melt_counterfactuals = df_hybrid_trend_weighted_melt.where(df_hybrid_trend_weighted_melt['value'] <= DS_historical_hybrid_weighted['Yield'].min().values )

df_hybrid_trend_weighted_melt_counterfactuals_split = df_hybrid_trend_weighted_melt_counterfactuals[df_hybrid_trend_weighted_melt_counterfactuals['value'] > - 10].copy()
df_hybrid_trend_weighted_melt_counterfactuals_split['RCP'] = df_hybrid_trend_weighted_melt_counterfactuals_split.variable.str.split('_').str[-1]

print("Number of impact analogues:", (df_hybrid_weighted_melt_counterfactuals['value'] > -10).sum(),
      'An average per scenario of',(df_hybrid_weighted_melt_counterfactuals['value'] > -10).sum()/ (len(df_hybrid_weighted_melt_counterfactuals.variable.unique())-1) )

years_counterfactuals = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > -10]

plt.figure(figsize=(8,5), dpi=300) #plot clusters
plt.axhline(y = DS_historical_hybrid_weighted['Yield'].min().values)
sns.scatterplot(data = df_hybrid_weighted_melt_counterfactuals, 
                x = df_hybrid_weighted_melt_counterfactuals.index, y='value', hue = 'variable')


plt.figure(figsize = (8,6),dpi=200)
bar = sns.boxplot(data = df_hybrid_weighted_melt_counterfactuals_split, y= df_hybrid_weighted_melt_counterfactuals_split['value'] - DS_historical_hybrid_weighted['Yield'].min().values , x= 'RCP')
plt.title("Counterfactuals with reference to the factual event")
plt.ylabel('Yield anomaly (ton/ha)')
plt.show()

plt.figure(figsize = (8,6),dpi=200)
sns.histplot( x = df_hybrid_weighted_melt_counterfactuals_split[['RCP','value']]['RCP'].values)
plt.title("Number of counterfactuals per RCP")
plt.show()

plt.figure(figsize = (8,6),dpi=200)
sns.histplot( x = df_hybrid_weighted_melt_counterfactuals_split[['RCP','value']]['value'].values, hue = df_hybrid_weighted_melt_counterfactuals_split[['RCP','value']]['RCP'].values)
plt.title("Distributions of counterfactulas per RCP")
plt.show()

#%% Figure 2 - timeseries with factual baseline

# TREND
plt.figure(figsize=(8,6), dpi=300) #plot clusters
plt.axhline(y = DS_historical_hybrid_weighted['Yield'].min().values, linestyle = 'dashed', label = '2012 event')
DS_hybrid_trend_all_weighted['gfdl_26'].plot(label = 'GFDL-esm4 SSP126',marker='o', color='tab:blue')
DS_hybrid_trend_all_weighted['gfdl_85'].plot(label = 'GFDL-esm4 SSP585',marker='o', color='tab:orange')
DS_hybrid_trend_all_weighted['ipsl_26'].plot(label = 'IPSL-cm6a-lr SSP126',marker='^', color='tab:blue')
DS_hybrid_trend_all_weighted['ipsl_85'].plot(label = 'IPSL-cm6a-lr SSP585',marker='^', color='tab:orange')
DS_hybrid_trend_all_weighted['ukesm_26'].plot(label = 'UKESM1-0-ll SSP126',marker='s', color='tab:blue')
DS_hybrid_trend_all_weighted['ukesm_85'].plot(label = 'UKESM1-0-ll SSP585',marker='s', color='tab:orange')
plt.title("a) Yield timeseries in no-adaptation scenario")
plt.legend()
plt.ylim(1,3)
plt.ylabel('Yield (ton/ha)')
plt.tight_layout()
plt.show()

plt.figure(figsize = (8,6), dpi=200)
bar = sns.boxplot(data = df_hybrid_trend_weighted_melt_counterfactuals_split, y= df_hybrid_trend_weighted_melt_counterfactuals_split['value'] - DS_historical_hybrid_weighted['Yield'].min().values , x= 'RCP')
plt.title("Trended counterfactuals with reference to the factual event")
plt.ylabel('Yield anomaly (ton/ha)')
plt.ylim(-1.2,0.2)
plt.show()

# 
plt.figure(figsize = (8,6), dpi=200)
sns.histplot( x = df_hybrid_trend_weighted_melt_counterfactuals_split[['RCP','value']]['RCP'].sort_values().values)
plt.title("Trended counterfactuals per RCP")
plt.ylim(0,90)
plt.show()


# DETRENDED
plt.figure(figsize=(8,6), dpi=300) #plot clusters
plt.axhline(y = DS_historical_hybrid_weighted['Yield'].min().values, linestyle = 'dashed', label = '2012 event')
DS_hybrid_gfdl_26_weighted['Yield'].plot(label = 'GFDL-esm4 SSP126',marker='o', color='tab:blue')
DS_hybrid_gfdl_85_weighted['Yield'].plot(label = 'GFDL-esm4 SSP585',marker='o', color='tab:orange')
DS_hybrid_ipsl_26_weighted['Yield'].plot(label = 'IPSL-cm6a-lr SSP126',marker='^', color='tab:blue')
DS_hybrid_ipsl_85_weighted['Yield'].plot(label = 'IPSL-cm6a-lr SSP585',marker='^', color='tab:orange')
DS_hybrid_ukesm_26_weighted['Yield'].plot(label = 'UKESM1-0-ll SSP126',marker='s', color='tab:blue')
DS_hybrid_ukesm_85_weighted['Yield'].plot(label = 'UKESM1-0-ll SSP585',marker='s', color='tab:orange')
plt.title("b) Yield timeseries in full-adaptation scenario")
plt.legend()
plt.ylim(1,3)
plt.ylabel('Yield (ton/ha)')
plt.tight_layout()
plt.show()

plt.figure(figsize = (8,6), dpi=200)
bar = sns.boxplot(data = df_hybrid_weighted_melt_counterfactuals_split, y= df_hybrid_weighted_melt_counterfactuals_split['value'] - DS_historical_hybrid_weighted['Yield'].min().values , x= 'RCP')
plt.title("Detrended counterfactuals with reference to the factual event")
plt.ylabel('Yield anomaly (ton/ha)')
plt.ylim(-1.2,0.2)
plt.show()

plt.figure(figsize = (8,6), dpi=200)
sns.histplot( x = df_hybrid_weighted_melt_counterfactuals_split[['RCP','value']]['RCP'].sort_values().values)
plt.title("Detrended counterfactuals per RCP")
plt.ylim(0,90)
plt.show()


#%% Comparison of factual and counterfactuals

# =============================================================================
# # Identify the FACTUAL case - Minimum value with respect to the mean
# =============================================================================
DS_anomaly_factual = DS_historical_hybrid['Yield'].sel(time = DS_historical_hybrid_weighted['Yield'].idxmin()) - DS_historical_hybrid['Yield'].mean(['time'])
# Identify weighted values of the factual cases
yield_factual = DS_historical_hybrid_weighted.to_dataframe().min().values
# Plot spatial maps - mean values
plot_2d_am_map(DS_historical_hybrid['Yield'].mean(['time']))
plot_2d_am_map(DS_historical_hybrid['Yield'].sel(time = 2010), colormap = 'Blues')
# Plot factual shock
plot_2d_am_map(DS_anomaly_factual, colormap = 'RdBu', vmin = -1, vmax=1)
'''
We see here that the failure took place in all three countries.
'''

# =============================================================================
# COUNTERFACTUALS
# =============================================================================
# Find impact analogues based on the yield value of the factual case.
DS_counterfactuals_weighted_am = DS_hybrid_all_weighted.where(DS_hybrid_all_weighted <= yield_factual)
print("Number of impact analogues per scenario:", (DS_counterfactuals_weighted_am > -10).sum())
# Check years of counterfactuals
list_counterfactuals_scenarios = []
for feature in list(DS_counterfactuals_weighted_am.keys()):
    feature_counterfactuals = DS_counterfactuals_weighted_am[feature].dropna(dim = 'time')
    print(feature_counterfactuals.time.values)
    list_counterfactuals_scenarios.append(feature_counterfactuals.time.values)
print('Total counterfactuals: ', len(np.hstack(list_counterfactuals_scenarios)) )

# Create dataset with spatailly explict counterfactuals only 
DS_counterfactuals_spatial = DS_hybrid_all.where(DS_counterfactuals_weighted_am > -10)

# Find the counterfactual shocks using a baseline as reference, either historical yields or the factual as reference
DS_hybrid_counterfactuals_spatial_shock = DS_counterfactuals_spatial.dropna('time', how='all') - DS_historical_hybrid['Yield'].mean('time')
DS_hybrid_counterfactuals_spatial_shock_2012 = DS_counterfactuals_spatial.dropna('time', how='all') - DS_historical_hybrid['Yield'].sel(time = 2012)

# =============================================================================
# # Plots the counterfactuals per scenario 
# =============================================================================
for feature in list(DS_hybrid_counterfactuals_spatial_shock.keys()):
    plot_2d_am_multi(DS_hybrid_counterfactuals_spatial_shock[feature].sel(time = DS_counterfactuals_weighted_am[feature].time.where(DS_counterfactuals_weighted_am[feature] > -10).dropna(dim = 'time')), map_title = feature )

# for feature in list(DS_hybrid_counterfactuals_spatial_shock_2012.keys()):
#     plot_2d_am_multi(DS_hybrid_counterfactuals_spatial_shock_2012[feature].sel(time = DS_counterfactuals_weighted_am[feature].time.where(DS_counterfactuals_weighted_am[feature] > -10).dropna(dim = 'time')), map_title = feature )
   
# =============================================================================
# Climatic analysis of counterfactuals
# =============================================================================
# Function converting the climate data for the counterfactuals into weighted timeseries 
def convert_clim_weighted_ensemble(df_clim, DS_counterfactuals_weighted_country, feature, DS_area):
    DS_clim = xr.Dataset.from_dataframe(df_clim)
    DS_clim = rearrange_latlot(DS_clim)
    # Countefactuals only
    DS_clim_counter = DS_clim.where(DS_counterfactuals_weighted_country[feature].time.where(DS_counterfactuals_weighted_country[feature] > -10).dropna(dim = 'time'))
   
    DS_clim_counter_weight_country = weighted_conversion(DS_clim_counter, DS_area = DS_area)
    df_clim_counter_weight_country = DS_clim_counter_weight_country.to_dataframe()
    df_clim_counter_weight_country['scenario'] = 'Analogues'
    df_clim_counter_weight_country['model_used'] = feature
    
    return df_clim_counter_weight_country

# Load historical climatic information - input
df_hybrid_am = pd.read_csv('dataset_input_hybrid_am_forML.csv', index_col=[0,1,2],).iloc[:, 1:].copy()
# Plot the mean temperature values for all counterfactuals and subtract from the 2012 year.
DS_conditions_hist = xr.Dataset.from_dataframe(df_hybrid_am)
DS_conditions_hist = rearrange_latlot(DS_conditions_hist)

# Load clim projections and Remove yield information
df_hybrid_fut_ukesm_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_ukesm1-0-ll_ssp585_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_ukesm_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_ukesm1-0-ll_ssp126_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_gfdl_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_gfdl-esm4_ssp585_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_gfdl_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_gfdl-esm4_ssp126_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_ipsl_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_ipsl-cm6a-lr_ssp585_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_ipsl_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_ipsl-cm6a-lr_ssp126_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]

# Conversion of historical series to weighted timeseries    
DS_conditions_hist_weighted_am = weighted_conversion(DS_conditions_hist, DS_area = DS_harvest_area_hist)
DS_conditions_2012_weighted_am = DS_conditions_hist_weighted_am.sel(time=2012)
# DS to df
df_clim_hist_weighted = DS_conditions_hist_weighted_am.to_dataframe()
df_clim_hist_weighted['scenario'] = 'Hist'
df_clim_hist_weighted['model_used'] = 'Hist'

# Conversion of future series to weighted timeseries    
df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_am, 'ukesm_85', DS_harvest_area_fut)    
df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_am, 'ukesm_26', DS_harvest_area_fut)    
df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_am, 'gfdl_85', DS_harvest_area_fut)    
df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_am, 'gfdl_26', DS_harvest_area_fut)    
df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_am, 'ipsl_85', DS_harvest_area_fut)    
df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_am, 'ipsl_26', DS_harvest_area_fut)    

# Merge dataframes with different names
df_clim_counterfactuals_weighted_all_am = pd.concat([df_clim_hist_weighted, df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
                                                  df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
                                                  df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])


# =============================================================================
# # Plot boxplots comparing the historical events, the 2012 event and the counterfactuals
# =============================================================================
names = df_clim_counterfactuals_weighted_all_am.columns.drop(['scenario', 'model_used'])
ncols = len(names)
fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am.columns)/3)), figsize=(10, 8), dpi=300, gridspec_kw=dict(height_ratios=[1,1]))

for name, ax in zip(names, axes.flatten()):
    df_merge_subset_am = df_clim_counterfactuals_weighted_all_am[df_clim_counterfactuals_weighted_all_am.index != 2012].loc[:,[name,'scenario']]
    df_merge_subset_am['variable'] = name
    g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Hist').dropna(), orient='v', ax=ax)
    g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
    ax.axhline( y = DS_conditions_2012_weighted_am[name].mean(), color = 'red', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
    g1.set(xticklabels=[])  # remove the tick labels
    g1.set(xlabel= name)
    if name in names[0:3]:
        g1.set(ylabel= 'Precipitation (mm/month)')  # remove the axis label  
    elif name in names[3:6]:
        g1.set(ylabel='Temperature (°C)' )  # remove the axis label   
    ax.get_legend().remove()
    g1.tick_params(bottom=False)  # remove the ticks
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=[0.3,0], ncol=3, frameon=False)
plt.suptitle('Counterfactuals for Americas')
plt.tight_layout()
plt.show()

#%% Assessment of losses per country 
# =============================================================================
'''
Try to compare the counterfatuals from a country perspective, does it mean one of the three countries is more prone to losses / failures?
'''
# Determine country level yields for historical period
DS_historical_hybrid_us = xr.load_dataset("output_models_am/hybrid_epic_us-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
DS_historical_hybrid_br = xr.load_dataset("output_models_am/hybrid_epic_br-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
DS_historical_hybrid_arg = xr.load_dataset("output_models_am/hybrid_epic_arg-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")

DS_mirca_us_hist = DS_harvest_area_sim.where(DS_historical_hybrid_us['Yield'] > -10)
DS_mirca_br_hist = DS_harvest_area_sim.where(DS_historical_hybrid_br['Yield'] > -10)
DS_mirca_arg_hist = DS_harvest_area_sim.where(DS_historical_hybrid_arg['Yield'] > -10)

plot_2d_am_map(DS_historical_hybrid_us.Yield.sel(time=1999))

# Determine country level yields for future projections
DS_counterfactual_us = DS_counterfactuals_spatial.where(DS_historical_hybrid_us['Yield'].mean('time') > -10)
DS_mirca_us = DS_harvest_area_fut.where(DS_historical_hybrid_us['Yield'].mean('time') > -10)

DS_counterfactual_br = DS_counterfactuals_spatial.where(DS_historical_hybrid_br['Yield'].mean('time') > -10)
DS_mirca_br = DS_harvest_area_fut.where(DS_historical_hybrid_br['Yield'].mean('time') > -10)

DS_counterfactual_arg = DS_counterfactuals_spatial.where(DS_historical_hybrid_arg['Yield'].mean('time') > -10)
DS_mirca_arg = DS_harvest_area_fut.where(DS_historical_hybrid_arg['Yield'].mean('time') > -10)

plot_2d_am_map(DS_counterfactual_us.ukesm_85.sel(time=2042))
plot_2d_am_map(DS_counterfactual_br.ukesm_85.sel(time=2042))
plot_2d_am_map(DS_counterfactual_arg.ukesm_85.sel(time=2042))
plot_2d_am_map(DS_mirca_arg.harvest_area)

# Weighted analysis historical
DS_historical_hybrid_us_weight = weighted_conversion(DS_historical_hybrid_us, DS_area = DS_mirca_us_hist)
DS_historical_hybrid_br_weight = weighted_conversion(DS_historical_hybrid_br, DS_area = DS_mirca_br_hist)
DS_historical_hybrid_arg_weight = weighted_conversion(DS_historical_hybrid_arg, DS_area = DS_mirca_arg_hist)

# Plot historical timeline of weighted soybean yield
plt.plot(DS_historical_hybrid_us_weight.time, DS_historical_hybrid_us_weight['Yield'], label = 'US')
plt.plot(DS_historical_hybrid_br_weight.time, DS_historical_hybrid_br_weight['Yield'], label = 'BR')
plt.plot(DS_historical_hybrid_arg_weight.time, DS_historical_hybrid_arg_weight['Yield'], label = 'ARG')
plt.title('Historical hybrid data')
plt.ylim(1,3)
plt.legend()
plt.show()

# Plot aggregated historical timeseries - equivalent to production / global contribution
DS_mirca_us['harvest_area'].sum() + DS_mirca_br['harvest_area'].sum() + DS_mirca_arg['harvest_area'].sum() - DS_harvest_area_hist.mean('time')['harvest_area'].sum()

# Plot production - TEST
def production(DS, DS_area):
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'] ) ).to_dataset(name = 'Yield')
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = ((DS * DS_area['harvest_area'] ) )
    return DS_weighted.sum(['lat','lon'])

DS_produc_am = production(DS_historical_hybrid, DS_harvest_area_fut)
DS_produc_us = production(DS_historical_hybrid_us, DS_mirca_us_hist)
DS_produc_br = production(DS_historical_hybrid_br, DS_mirca_br_hist)
DS_produc_arg = production(DS_historical_hybrid_arg, DS_mirca_arg_hist)

plt.plot(DS_produc_am.time, DS_produc_am['Yield'], label = 'AM')
plt.stackplot(DS_produc_us.time, DS_produc_us['Yield'],
              DS_produc_br['Yield'], DS_produc_arg['Yield'], labels = ['US', 'BR', 'ARG'])
plt.legend()
plt.tight_layout()
plt.show()

# Weighted analysis future projections
DS_counterfactual_us_weighted = weighted_conversion(DS_counterfactual_us, DS_area = DS_mirca_us)
DS_counterfactual_us_weighted = DS_counterfactual_us_weighted.where(DS_counterfactual_us_weighted > 0).dropna('time', how = 'all')

DS_counterfactual_br_weighted = weighted_conversion(DS_counterfactual_br, DS_area = DS_mirca_br)
DS_counterfactual_br_weighted = DS_counterfactual_br_weighted.where(DS_counterfactual_br_weighted > 0).dropna('time', how = 'all')

DS_counterfactual_arg_weighted = weighted_conversion(DS_counterfactual_arg, DS_area = DS_mirca_arg)
DS_counterfactual_arg_weighted = DS_counterfactual_arg_weighted.where(DS_counterfactual_arg_weighted > 0).dropna('time', how = 'all')

df_mean_yields = pd.DataFrame( [DS_counterfactual_us_weighted.to_dataframe().mean(), DS_counterfactual_br_weighted.to_dataframe().mean(),
                               DS_counterfactual_arg_weighted.to_dataframe().mean()], 
                                          index = ['US', 'BR', 'ARG'])

# =============================================================================
# # Plot accumulated mean losses of projected counterfactuals per country with reference to aggregated series
# =============================================================================
df_mean_yields.plot.bar(figsize = (12,8))
plt.axhline(y = yield_factual, linestyle = 'dashed', label = 'Factual')
plt.show()

# Plot accumulated mean losses per country - Absolute shock
sns.barplot(data = df_mean_yields.T )
plt.axhline(y = yield_factual, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend()
plt.title('Counterfactuals absolute values')
plt.ylabel('ton/ha')
plt.show()

# Plot accumulated mean losses per country - Absolute shock
sns.barplot(data = df_mean_yields.T - DS_historical_hybrid_weighted['Yield'].mean().values)
plt.axhline(y = yield_factual - DS_historical_hybrid_weighted['Yield'].mean().values, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend()
plt.title('Analogues of the 2012 event in each country')
plt.ylabel('Yield anomaly (ton/ha)')
plt.show()

# Plot accumulated mean losses per country - Relative shock
sns.barplot(data = df_mean_yields.T / DS_historical_hybrid_weighted['Yield'].mean().values - 1)
plt.axhline(y = yield_factual / DS_historical_hybrid_weighted['Yield'].mean().values - 1, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend()
plt.title('Counterfactuals relative shock')
plt.show()


#%% Analysis at a country level - 2012 event from a country perspective, aggregation at country level
# =============================================================================
# Failures per country
yield_factual_2012_us = DS_historical_hybrid_us_weight.sel(time = 2012).Yield.values
yield_factual_2012_br = DS_historical_hybrid_br_weight.sel(time = 2012).Yield.values
yield_factual_2012_arg = DS_historical_hybrid_arg_weight.sel(time = 2012).Yield.values
yield_factual_2012_am = pd.DataFrame([yield_factual_2012_us, yield_factual_2012_br, yield_factual_2012_arg], index = ['US', 'BR', 'ARG'])

# =============================================================================
# # Historical failures at a country level
# =============================================================================
plt.bar(x = 'US', height = yield_factual_2012_us)
plt.bar(x = 'BR', height = yield_factual_2012_br)
plt.bar(x = 'ARG', height = yield_factual_2012_arg)
plt.axhline(y = yield_factual, color = 'black', linestyle = 'dashed', label = 'Aggregated')
plt.title('2012 yield shock')
plt.legend()
plt.ylabel('ton/ha')
plt.show()

# Absolute shocks
plt.bar(x = 'US', height = yield_factual_2012_us - DS_historical_hybrid_us_weight.Yield.mean('time').values)
plt.bar(x = 'BR', height = yield_factual_2012_br - DS_historical_hybrid_br_weight.Yield.mean('time').values)
plt.bar(x = 'ARG', height = yield_factual_2012_arg- DS_historical_hybrid_arg_weight.Yield.mean('time').values)
plt.axhline(y = yield_factual - DS_historical_hybrid_weighted.Yield.mean('time').values, color = 'black', linestyle = 'dashed', label = 'Aggregated')
plt.title('2012 anomaly absolute shock')
plt.ylabel('Yield anomaly (ton/ha)')
plt.legend()
plt.show()

# Relative shocks
plt.bar(x = 'US', height = yield_factual_2012_us / DS_historical_hybrid_us_weight.Yield.mean('time').values - 1)
plt.bar(x = 'BR', height = yield_factual_2012_br / DS_historical_hybrid_br_weight.Yield.mean('time').values - 1)
plt.bar(x = 'ARG', height = yield_factual_2012_arg / DS_historical_hybrid_arg_weight.Yield.mean('time').values - 1)
plt.axhline(y = yield_factual / DS_historical_hybrid_weighted.Yield.mean('time').values - 1, color = 'black', linestyle = 'dashed', label = 'Aggregated')
plt.title('2012 anomaly relative shock')
plt.ylabel('0-1')
plt.legend()
plt.show()

# =============================================================================
# Projections shocks with respect to historical country levels
# =============================================================================
mean_values_per_country = pd.DataFrame([DS_historical_hybrid_us_weight.Yield.mean('time').values, DS_historical_hybrid_br_weight.Yield.mean('time').values,  DS_historical_hybrid_arg_weight.Yield.mean('time').values], index = ['US', 'BR', 'ARG'])

# Counterfactuals of 2012 event in absolute values
sns.barplot(data = (df_mean_yields.T ))
plt.axhline(y = yield_factual, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend()
plt.title('Counterfactuals absolute values')
plt.ylabel('ton/ha')
plt.show()

# Counterfactuals of 2012 event in absolute shock
sns.barplot(data = (df_mean_yields.T - mean_values_per_country.T.values))
plt.axhline(y = yield_factual - DS_historical_hybrid_weighted['Yield'].mean().values, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend()
plt.title('Local analogues of each country')
plt.ylabel('ton/ha')
plt.show()

# Plot accumulated mean losses per country - Relative shock
sns.barplot(data = (df_mean_yields.T / mean_values_per_country.T.values - 1))
plt.axhline(y = yield_factual / DS_historical_hybrid_weighted['Yield'].mean().values - 1, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend()
plt.title('Counterfactuals relative shock')
plt.show()

#%% Number of counterfactuals per country level, which scenarios and years they appear.
REDO THIS ONE - recheck all the information on the counterfactuals




# Local counterfactuals
def counterfactual_generation(DS_yields, DS_mirca_country, local_counterfactual):
    DS_hybrid_country = DS_yields.where(DS_mirca_country['harvest_area'] > 0)
    DS_projections_weighted_country = weighted_conversion(DS_hybrid_country, DS_area = DS_mirca_country)
    DS_projections_weighted_country_counterfactual = DS_projections_weighted_country.where(DS_projections_weighted_country <= local_counterfactual).dropna('time', how = 'all')
    return DS_projections_weighted_country_counterfactual

def counterfactuals_country_level(DS_yields, DS_mirca_country, DS_historical_hybrid_country):
    DS_projections_country = DS_yields.where(DS_mirca_country['harvest_area'].mean() > 0)
    
    DS_historical_hybrid_country_weight = weighted_conversion(DS_historical_hybrid_country, DS_area = DS_mirca_country)
    yield_factual_2012 = DS_historical_hybrid_country_weight.sel(time = 2012).Yield.values
    
    DS_projections_country_weighted = weighted_conversion(DS_projections_country, DS_area = DS_mirca_country.mean('time'))
    DS_projections_country_weighted = DS_projections_country_weighted.where(DS_projections_country_weighted > 0).dropna('time', how = 'all')
    
    DS_counterfactuals_weighted_country = DS_projections_country_weighted.where(DS_projections_country_weighted <= yield_factual_2012)
    print("Number of impact analogues per scenario for the country:", (DS_counterfactuals_weighted_country > -10).sum())
    
    # Check years of counterfactuals
    list_counterfactuals_scenarios = []
    for feature in list(DS_counterfactuals_weighted_country.keys()):
        feature_counterfactuals = DS_counterfactuals_weighted_country[feature].dropna(dim = 'time')
        print(feature_counterfactuals.time.values)
        list_counterfactuals_scenarios.append(feature_counterfactuals.time.values)
        number_counter = len(np.hstack(list_counterfactuals_scenarios)) 
    print('total counterfactuals: ', number_counter)
        
    # Create dataset with spatailly explict counterfactuals only 
    DS_counterfactuals_spatial_country = DS_projections_country.where(DS_counterfactuals_weighted_country > -10)

    # Find the counterfactual shocks using a baseline as reference, either historical yields or the factual as reference
    DS_hybrid_counterfactuals_spatial_shock_country = DS_counterfactuals_spatial_country.dropna('time', how='all') - DS_historical_hybrid_country['Yield'].mean('time')
    # DS_hybrid_counterfactuals_spatial_shock_country_2012 = DS_counterfactuals_spatial_country.dropna('time', how='all') - DS_historical_hybrid_country['Yield'].sel(time = 2012)

    # =============================================================================
    # # Plots the counterfactuals per scenario 
    # =============================================================================
    for feature in list(DS_hybrid_counterfactuals_spatial_shock_country.keys()):
        plot_2d_am_multi(DS_hybrid_counterfactuals_spatial_shock_country[feature].sel(time = DS_counterfactuals_weighted_country[feature].time.where(DS_counterfactuals_weighted_country[feature] > -10).dropna(dim = 'time')), map_title = feature )
    
    # for feature in list(DS_hybrid_counterfactuals_spatial_shock_country_2012.keys()):
    #     plot_2d_am_multi(DS_hybrid_counterfactuals_spatial_shock_country_2012[feature].sel(time = DS_counterfactuals_weighted_country[feature].time.where(DS_counterfactuals_weighted_country[feature] > -10).dropna(dim = 'time')), map_title = feature )
    return DS_counterfactuals_weighted_country, number_counter

DS_projections_weighted_us_counterfactual = counterfactual_generation(DS_hybrid_all, DS_mirca_us, yield_factual_2012_us)
DS_projections_weighted_br_counterfactual = counterfactual_generation(DS_hybrid_all, DS_mirca_br, yield_factual_2012_br)
DS_projections_weighted_arg_counterfactual = counterfactual_generation(DS_hybrid_all, DS_mirca_arg, yield_factual_2012_arg)
    
DS_counterfactuals_weighted_us, number_counter_us = counterfactuals_country_level(DS_hybrid_all, DS_mirca_us_hist, DS_historical_hybrid_us)
DS_counterfactuals_weighted_br, number_counter_br = counterfactuals_country_level(DS_hybrid_all, DS_mirca_br_hist, DS_historical_hybrid_br)
DS_counterfactuals_weighted_arg, number_counter_arg = counterfactuals_country_level(DS_hybrid_all, DS_mirca_arg_hist, DS_historical_hybrid_arg)

plt.bar(x = ['US','BR','ARG'], height = [number_counter_us, number_counter_br, number_counter_arg])
plt.title('Number of local counterfactuals per country')
plt.show()

years_counterfactuals_am = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > -10]

years_counterfactuals_us = DS_counterfactuals_weighted_us.to_dataframe().melt(ignore_index = False).dropna()
years_counterfactuals_br = DS_counterfactuals_weighted_br.to_dataframe().melt(ignore_index = False).dropna()
years_counterfactuals_arg = DS_counterfactuals_weighted_arg.to_dataframe().melt(ignore_index = False).dropna()

# TRIAL 2 - check for each scenario the times we see failures in each country and overall
for feature in years_counterfactuals_am.variable:
    data = years_counterfactuals_am.loc[years_counterfactuals_am['variable'] == feature]
    data_us = years_counterfactuals_us.loc[years_counterfactuals_us['variable'] == feature]
    data_br = years_counterfactuals_br.loc[years_counterfactuals_br['variable'] == feature]
    data_arg = years_counterfactuals_arg.loc[years_counterfactuals_arg['variable'] == feature]
    sns.scatterplot(data = data, x = data.index, y=data['value'], label = 'AM')
    sns.scatterplot(data = data_us, x = data_us.index, y=data_us['value'], label = 'US')
    sns.scatterplot(data = data_br, x = data_br.index, y=data_br['value'], label = 'BR')
    sns.scatterplot(data = data_arg, x = data_arg.index, y=data_arg['value'], label = 'ARG')
    plt.legend()
    plt.show()


#%% Yield - Climate interaction

# =============================================================================
# What are the climatic conditions leading to the failures? 
# =============================================================================


# =============================================================================
# # Comparing the weather conditions between historic times, 2012 and counterfactuals for each climatic variables and for each country
# =============================================================================

for country in ['US','BR','ARG']:
    if country == 'US':
        DS_area_hist = DS_mirca_us_hist
        DS_area_fut = DS_mirca_us
        DS_counterfactuals_weighted_country = DS_counterfactuals_weighted_us
    elif country == 'BR':
        DS_area_hist = DS_mirca_br_hist
        DS_area_fut = DS_mirca_br
        DS_counterfactuals_weighted_country = DS_counterfactuals_weighted_br
    elif country == 'ARG':
        DS_area_hist = DS_mirca_arg_hist
        DS_area_fut = DS_mirca_arg
        DS_counterfactuals_weighted_country = DS_counterfactuals_weighted_arg
   
    # Conversion of historical series to weighted timeseries    
    DS_conditions_hist_weighted_country = weighted_conversion(DS_conditions_hist, DS_area = DS_area_hist)
    DS_conditions_2012_weighted_country = DS_conditions_hist_weighted_country.sel(time=2012)
    # DS to df
    df_clim_hist_weighted = DS_conditions_hist_weighted_country.to_dataframe()
    df_clim_hist_weighted['scenario'] = 'Hist'
    df_clim_hist_weighted['model_used'] = 'Hist'
    
    # Conversion of future series to weighted timeseries    
    df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_country, 'ukesm_85', DS_area_fut)    
    df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_country, 'ukesm_26', DS_area_fut)    
    df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_country, 'gfdl_85', DS_area_fut)    
    df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_country, 'gfdl_26', DS_area_fut)    
    df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_country, 'ipsl_85', DS_area_fut)    
    df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_country, 'ipsl_26', DS_area_fut)    
    
    # Merge dataframes with different names
    df_clim_counterfactuals_weighted_all = pd.concat([df_clim_hist_weighted, df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
                                                      df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
                                                      df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])
    
# =============================================================================
#     # Plot boxplots comparing the historical events, the 2012 event and the counterfactuals
# =============================================================================
    names = df_clim_counterfactuals_weighted_all.columns.drop(['scenario', 'model_used'])
    ncols = len(names)
    fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all.columns)/3)), figsize=(10, 8), dpi=300, gridspec_kw=dict(height_ratios=[1,1]))
    
    for name, ax in zip(names, axes.flatten()):
        df_merge_subset = df_clim_counterfactuals_weighted_all[df_clim_counterfactuals_weighted_all.index != 2012].loc[:,[name,'scenario']]
        df_merge_subset['variable'] = name
        g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset, orient='v', ax=ax)
        # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
        ax.axhline( y = DS_conditions_2012_weighted_country[name].mean(), color = 'gray', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
        g1.set(xticklabels=[])  # remove the tick labels
        g1.set(xlabel= name)
        if name in names[0:3]:
            g1.set(ylabel= 'Precipitation (mm/month)')  # remove the axis label  
        elif name in names[3:6]:
            g1.set(ylabel='Temperature (°C)' )  # remove the axis label   
        ax.get_legend().remove()
        g1.tick_params(bottom=False)  # remove the ticks
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=[0.3,0], ncol=3, frameon=False)
    plt.suptitle(f'Counterfactuals for {country} region')
    plt.tight_layout()
    plt.show()

#%% Test figure values pdp

df_hybrid_am_test = pd.read_csv('dataset_input_hybrid_am_forML.csv', index_col=[0,1,2],).copy()
DS_conditions_hist_weighted_am = weighted_conversion(df_hybrid_am_test.to_xarray(), DS_area = DS_harvest_area_hist)

features_to_plot = [4,5,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp6 = PartialDependenceDisplay.from_estimator(model_to_be_used, DS_conditions_hist_weighted_am.to_dataframe(), features_to_plot, pd_line_kw={'color':'k'},percentiles=(0.01,0.99), ax = ax1)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
# disp5.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Extrapolation", "color": "red"})
disp6.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(1, 3)
ax2.set_ylim(1, 3)
ax3.set_ylim(1, 3)
ax1.scatter(x=df_clim_counterfactuals_weighted_all_am.loc[df_clim_counterfactuals_weighted_all_am['scenario'] == 'Analogues'].loc[:,'txm_3'].sort_values().values, y = DS_counterfactuals_weighted_am.to_dataframe().melt(ignore_index = False)['value'].dropna().sort_values())
ax2.scatter(x=df_clim_counterfactuals_weighted_all_am.loc[df_clim_counterfactuals_weighted_all_am['scenario'] == 'Analogues'].loc[:,'txm_4'].sort_values().values, y = DS_counterfactuals_weighted_am.to_dataframe().melt(ignore_index = False)['value'].dropna().sort_values())
ax3.scatter(x=df_clim_counterfactuals_weighted_all_am.loc[df_clim_counterfactuals_weighted_all_am['scenario'] == 'Analogues'].loc[:,'txm_5'].sort_values().values, y = DS_counterfactuals_weighted_am.to_dataframe().melt(ignore_index = False)['value'].dropna().sort_values())
plt.setp(disp6.deciles_vlines_, visible=False)
plt.setp(disp6.deciles_vlines_, visible=False)
ax1.legend()
plt.show()



features_to_plot = [1,2,3]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp3 = PartialDependenceDisplay.from_estimator(model_to_be_used, DS_conditions_hist_weighted_am.to_dataframe(), features_to_plot, pd_line_kw={'color':'k'},percentiles=(0.01,0.99), ax = ax1)
plt.ylim(0, 2.6)
plt.setp(disp3.deciles_vlines_, visible=False)

fig, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(15, 8))
disp3.plot(ax=[ax4, ax5, ax6], line_kw={"label": "Historical", "color": "k"})
ax4.set_ylim(1, 3)
ax5.set_ylim(1, 3)
ax6.set_ylim(1, 3)
ax4.scatter(x=df_clim_counterfactuals_weighted_all_am.loc[df_clim_counterfactuals_weighted_all_am['scenario'] == 'Analogues'].loc[:,'prcptot_3'].sort_values().values, y = DS_counterfactuals_weighted_am.to_dataframe().melt(ignore_index = False)['value'].dropna().sort_values())
ax5.scatter(x=df_clim_counterfactuals_weighted_all_am.loc[df_clim_counterfactuals_weighted_all_am['scenario'] == 'Analogues'].loc[:,'prcptot_4'].sort_values().values, y = DS_counterfactuals_weighted_am.to_dataframe().melt(ignore_index = False)['value'].dropna().sort_values())
ax6.scatter(x=df_clim_counterfactuals_weighted_all_am.loc[df_clim_counterfactuals_weighted_all_am['scenario'] == 'Analogues'].loc[:,'prcptot_5'].sort_values().values, y = DS_counterfactuals_weighted_am.to_dataframe().melt(ignore_index = False)['value'].dropna().sort_values())
plt.setp(disp3.deciles_vlines_, visible=False)
# plt.setp(disp4.deciles_vlines_, visible=False)
ax1.legend()
plt.show()
######### TEST 2

#%% Trying to fit counterfactuals within the pdp of historical values
for country in ['US','BR','ARG']:
    if country == 'US':
        DS_area_hist = DS_mirca_us_hist
        DS_area_fut = DS_mirca_us
        DS_counterfactuals_weighted_country = DS_counterfactuals_weighted_us
    elif country == 'BR':
        DS_area_hist = DS_mirca_br_hist
        DS_area_fut = DS_mirca_br
        DS_counterfactuals_weighted_country = DS_counterfactuals_weighted_br
    elif country == 'ARG':
        DS_area_hist = DS_mirca_arg_hist
        DS_area_fut = DS_mirca_arg
        DS_counterfactuals_weighted_country = DS_counterfactuals_weighted_arg
        
    df_hybrid_am_test = pd.read_csv('dataset_input_hybrid_am_forML.csv', index_col=[0,1,2],).copy()
    DS_conditions_hist_weighted_country = weighted_conversion(df_hybrid_am_test.to_xarray(), DS_area = DS_area_hist)
   
    # DS to df
    df_clim_hist_weighted = DS_conditions_hist_weighted_country.to_dataframe()
    df_clim_hist_weighted['scenario'] = 'Hist'
    df_clim_hist_weighted['model_used'] = 'Hist'
    
    # Conversion of future series to weighted timeseries    
    df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_country, 'ukesm_85', DS_area_fut)    
    df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_country, 'ukesm_26', DS_area_fut)    
    df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_country, 'gfdl_85', DS_area_fut)    
    df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_country, 'gfdl_26', DS_area_fut)    
    df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_country, 'ipsl_85', DS_area_fut)    
    df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_country, 'ipsl_26', DS_area_fut)    
    
    # Merge dataframes with different names
    df_clim_counterfactuals_weighted_all_country = pd.concat([df_clim_hist_weighted, df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
                                                      df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
                                                      df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])
    
    df_yield_count_order = DS_counterfactuals_weighted_country.to_dataframe().melt(ignore_index = False).dropna().sort_values(axis = 'index', by= 'variable' )
    df_clim_order = df_clim_counterfactuals_weighted_all_country.loc[df_clim_counterfactuals_weighted_all_country['scenario'] == 'Analogues'].sort_values(axis = 'index', by= 'model_used' )
    
    features_to_plot = [4,5,6]
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
    disp6 = PartialDependenceDisplay.from_estimator(model_to_be_used, df_clim_hist_weighted.iloc[:,:-2], features_to_plot, pd_line_kw={'color':'k'},percentiles=(0.01,0.99), ax = ax1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    # disp5.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Extrapolation", "color": "red"})
    disp6.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Training", "color": "black"})
    ax1.set_ylim(1, 3)
    ax2.set_ylim(1, 3)
    ax3.set_ylim(1, 3)
    ax1.scatter(x= df_clim_order['txm_3'], y = df_yield_count_order['value'] )
    ax2.scatter(x= df_clim_order['txm_4'], y = df_yield_count_order['value'] )
    ax3.scatter(x= df_clim_order['txm_5'], y = df_yield_count_order['value'] )
    plt.setp(disp5.deciles_vlines_, visible=False)
    plt.setp(disp6.deciles_vlines_, visible=False)
    ax1.legend()
    plt.show()
    
    # PRECIPITATION
    features_to_plot = [1,2,3]
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
    disp6 = PartialDependenceDisplay.from_estimator(model_to_be_used, df_clim_hist_weighted.iloc[:,:-2], features_to_plot, pd_line_kw={'color':'k'},percentiles=(0.01,0.99), ax = ax1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    # disp5.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Extrapolation", "color": "red"})
    disp6.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Training", "color": "black"})
    ax1.set_ylim(1, 3)
    ax2.set_ylim(1, 3)
    ax3.set_ylim(1, 3)
    ax1.scatter(x= df_clim_order['prcptot_3'], y = df_yield_count_order['value'] )
    ax2.scatter(x= df_clim_order['prcptot_4'], y = df_yield_count_order['value'] )
    ax3.scatter(x= df_clim_order['prcptot_5'], y = df_yield_count_order['value'] )
    plt.setp(disp5.deciles_vlines_, visible=False)
    plt.setp(disp6.deciles_vlines_, visible=False)
    ax1.legend()
    plt.show()



#%% Random stuff

# for feature in list(DS_counterfactual_us.keys()):
#     plot_2d_am_multi(DS_counterfactual_us[feature].sel(time = DS_counterfactuals_weighted_am[feature].time.where(DS_counterfactuals_weighted_am[feature] > -10).dropna(dim = 'time')), map_title = feature )


# test = xr.load_dataset("soy_harvest_area_arg_1978_2019_05x05.nc")
# test['time'] = pd.date_range(start='1980', periods=test.sizes['time'], freq='YS').year

# plot_2d_am_map(test['harvest_area'].sel(time = 2000))

# test['harvest_area'].sel(time = 2000).plot()










