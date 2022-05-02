# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:05:38 2022

@author: morenodu
"""

from sklearn.model_selection import GridSearchCV
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

X_shuffle, y_shuffle = shuffle(X, y, random_state=0)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_shuffle, y_shuffle, test_size=0.1, random_state=0, shuffle = False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


batch_size = [2048, 1024,512]
epochs = [700]
neurons_list = [256,512] # Try 512, 256
learn_rate = [0.001, 0.005, 0.01]
dropout_train = [0.1, 0.2]
regul_values = [0, 0.00001]

list_results = []
for batch in batch_size:
    for epoch in epochs: 
        for neurons in neurons_list: 
            for lr in learn_rate: 
                for dropout_value in dropout_train:
                    for regul_value in regul_values:
                        #model
                        test_model = Sequential()
                        test_model.add(Dense(neurons, input_dim=len(X.columns), kernel_regularizer=regularizers.l2(regul_value))) 
                        test_model.add(BatchNormalization())
                        test_model.add(Activation('relu'))
                        test_model.add(Dropout(dropout_value))
        
                        test_model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regul_value)))
                        test_model.add(BatchNormalization())
                        test_model.add(Activation('relu'))
                        test_model.add(Dropout(dropout_value))
        
                        test_model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regul_value)))
                        test_model.add(BatchNormalization())
                        test_model.add(Activation('relu'))
                        test_model.add(Dropout(dropout_value))
        
                        test_model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regul_value)))
                        test_model.add(BatchNormalization())
                        test_model.add(Activation('relu'))
                        test_model.add(Dropout(dropout_value))
        
                        test_model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regul_value)))
                        test_model.add(BatchNormalization())
                        test_model.add(Activation('relu'))
                        test_model.add(Dropout(dropout_value))
        
                        test_model.add(Dense(1, activation='linear'))
                        
                        # compile the keras model
                        test_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate = lr), metrics=['mean_squared_error','mean_absolute_error'])
                        callback_model = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 50, restore_best_weights=True)
                        # mc = ModelCheckpoint('best_model_test_dnn2.model', monitor='val_loss', mode='min', save_best_only=True, verbose=0)
                        scikeras_regressor = Pipeline([
                            ('scaler', StandardScaler()),
                            ('estimator', KerasRegressor(model=test_model, batch_size = batch, epochs = epoch, validation_split= 0.1, random_state = 0, verbose=0, callbacks=[callback_model])) # validation_split= 0.1, callbacks=[callback_model_full, mc_full]
                        ])
                   
                        scikeras_regressor.fit(X_shuffle, y_shuffle)
                        
                        plt.figure(figsize=(16,8), dpi=250) #plot clusters
                        plt.plot(scikeras_regressor['estimator'].history_['loss'])
                        plt.plot(scikeras_regressor['estimator'].history_['val_loss'])
                        plt.title(f'batch: {batch} and epoch: {epoch} and neurons: {neurons} and learning rate : {lr}, dropout: {dropout_value},l2: {regul_value} ')
                        plt.ylabel('loss')
                        plt.xlabel('epoch')
                        plt.ylim(0, 0.2)
                        plt.legend(['train', 'test'], loc='upper left')
                        plt.show()
                        
                        best_epoch = np.argmin(scikeras_regressor['estimator'].history_['val_loss']) + 1
                        # Test performance
                        y_pred = scikeras_regressor.predict(X_test1)
                        
                        # report performance
                        print(f'Results for model: batch: {batch} and epoch: {epoch} and neurons: {neurons} and learning rate : {lr}, dropout: {dropout_value},l2: {regul_value} ')
                        print('Best epoch:', best_epoch )
                        print("R2 on test set:", round(r2_score(y_test1, y_pred),2))
                        print("Var score on test set:", round(explained_variance_score(y_test1, y_pred),2))
                        print("MAE on test set:", round(mean_absolute_error(y_test1, y_pred),5))
                        print("RMSE on test set:",round(mean_squared_error(y_test1, y_pred, squared=False),5))
                        list_results.append([batch, epoch, neurons, lr,dropout_value,regul_value,best_epoch, round(r2_score(y_test1, y_pred),2),round(mean_absolute_error(y_test1, y_pred),5) ])

df_results_config = pd.DataFrame(list_results, columns = ['batch', 'epoch', 'nodes', 'lr','dropout_value','regul_value','best_epoch','R2', 'MAE'])
df_results_config.to_csv('results_hyperparameter_dnn.csv')

#%%
TEXT: 
batch	epoch	nodes	lr	dropout_value	regul_value	best_epoch	R2	MAE
[2048, 700, 256, 0.001, 0.1, 0, 588, 0.7, 0.23249]
[2048, 700, 512, 0.001, 0.1, 1e-05, 347, 0.7, 0.23493]
[2048, 700, 512, 0.001, 0.1, 0.0001, 307, 0.68, 0.24476]
[2048, 700, 512, 0.001, 0.2, 1e-05, 402, 0.69, 0.23822]
[2048, 700, 512, 0.01, 0.1, 0, 403, 0.7, 0.23167]
105	1024	700	512	0.01	0.2	0.0	366	0.71	0.22949


Results for model: batch: 1024 and epoch: 512 and neurons: 256 and learning rate : 0.005, dropout: 0.1,l2: 0 
R2 on test set: 0.69
Var score on test set: 0.69
MAE on test set: 0.23901
RMSE on test set: 0.31914

Results for model: batch: 1024 and epoch: 512 and neurons: 256 and learning rate : 0.01, dropout: 0.1,l2: 0 
R2 on test set: 0.68
Var score on test set: 0.68
MAE on test set: 0.24479
RMSE on test set: 0.32426

Results for model: batch: 1024 and epoch: 512 and neurons: 512 and learning rate : 0.001, dropout: 0.1,l2: 0 
R2 on test set: 0.69
Var score on test set: 0.69
MAE on test set: 0.2367
RMSE on test set: 0.3185

Results for model: batch: 1024 and epoch: 700 and neurons: 512 and learning rate : 0.001, dropout: 0.1,l2: 1e-05 
Best epoch: 300
R2 on test set: 0.7
Var score on test set: 0.7
MAE on test set: 0.23468
RMSE on test set: 0.31481

Results for model: batch: 1024 and epoch: 700 and neurons: 512 and learning rate : 0.01, dropout: 0.2,l2: 0 
Best epoch: 366
R2 on test set: 0.71
Var score on test set: 0.71
MAE on test set: 0.22949
RMSE on test set: 0.31021

Results for model: batch: 512 and epoch: 700 and neurons: 256 and learning rate : 0.001, dropout: 0.1,l2: 0 
Best epoch: 308
R2 on test set: 0.7
Var score on test set: 0.7
MAE on test set: 0.23302
RMSE on test set: 0.3132

Results for model: batch: 512 and epoch: 700 and neurons: 256 and learning rate : 0.005, dropout: 0.1,l2: 0 
Best epoch: 271
R2 on test set: 0.7
Var score on test set: 0.7
MAE on test set: 0.23156
RMSE on test set: 0.31287

Results for model: batch: 512 and epoch: 700 and neurons: 256 and learning rate : 0.01, dropout: 0.1,l2: 0 
Best epoch: 297
R2 on test set: 0.7
Var score on test set: 0.7
MAE on test set: 0.23407
RMSE on test set: 0.31524
Results for model: batch: 512 and epoch: 700 and neurons: 512 and learning rate : 0.001, dropout: 0.2,l2: 0 
Best epoch: 404
R2 on test set: 0.7
Var score on test set: 0.71
MAE on test set: 0.22932
RMSE on test set: 0.31066

Results for model: batch: 512 and epoch: 700 and neurons: 512 and learning rate : 0.005, dropout: 0.2,l2: 0 
Best epoch: 382
R2 on test set: 0.7
Var score on test set: 0.7
MAE on test set: 0.23231
RMSE on test set: 0.31164

Results for model: batch: 512 and epoch: 700 and neurons: 512 and learning rate : 0.01, dropout: 0.2,l2: 0 
Best epoch: 326
R2 on test set: 0.7
Var score on test set: 0.7
MAE on test set: 0.23451
RMSE on test set: 0.31455

#%%
list_1 = pd.DataFrame([[1024, 700, 512, 0.01, 0.2, 0]])
list_2 = pd.DataFrame([[2048, 700, 512, 0.001, 0.2, 1e-05]])
list_3 = pd.DataFrame([[512, 700, 256, 0.001, 0.1, 0]])
list_4 = pd.DataFrame([[1024, 700, 512, 0.005, 0.1, 0]])
list_5 = pd.DataFrame([[2048, 700, 256, 0.001, 0.1, 0.0001]])
list_params = pd.concat([list_1, list_2, list_3, list_4, list_5], axis = 0)

best_epochs = []
for iteration in range(len(list_params)):
    batch, epoch, neurons, lr, dropout_value, regul_value = list_params.iloc[iteration]
    batch = int(batch)
    epoch = int(epoch)
    neurons = int(neurons)
    print(batch,epoch, neurons, lr, dropout_value, regul_value)
    
    #model
    test_model = Sequential()
    test_model.add(Dense(neurons, input_dim=len(X.columns), kernel_regularizer=regularizers.l2(regul_value))) 
    test_model.add(BatchNormalization())
    test_model.add(Activation('relu'))
    test_model.add(Dropout(dropout_value))
    
    test_model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regul_value)))
    test_model.add(BatchNormalization())
    test_model.add(Activation('relu'))
    test_model.add(Dropout(dropout_value))
    
    test_model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regul_value)))
    test_model.add(BatchNormalization())
    test_model.add(Activation('relu'))
    test_model.add(Dropout(dropout_value))
    
    test_model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regul_value)))
    test_model.add(BatchNormalization())
    test_model.add(Activation('relu'))
    test_model.add(Dropout(dropout_value))
    
    test_model.add(Dense(neurons, kernel_regularizer=regularizers.l2(regul_value)))
    test_model.add(BatchNormalization())
    test_model.add(Activation('relu'))
    test_model.add(Dropout(dropout_value))
    
    test_model.add(Dense(1, activation='linear'))
    
    # compile the keras model
    test_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate = lr), metrics=['mean_squared_error','mean_absolute_error'])
    callback_model = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 50, restore_best_weights=True)
    # mc = ModelCheckpoint('best_model_test_dnn2.model', monitor='val_loss', mode='min', save_best_only=True, verbose=0)
    scikeras_regressor = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', KerasRegressor(model=test_model, batch_size = batch, epochs = epoch, validation_split= 0.1, random_state = 0, verbose=0, callbacks=[callback_model])) # validation_split= 0.1, callbacks=[callback_model_full, mc_full]
    ])
    
    scikeras_regressor.fit(X_shuffle, y_shuffle)
    
    plt.figure(figsize=(14,8), dpi=250) #plot clusters
    plt.plot(scikeras_regressor['estimator'].history_['loss'])
    plt.plot(scikeras_regressor['estimator'].history_['val_loss'])
    plt.title(f'batch: {batch} and epoch: {epoch} and neurons: {neurons} and learning rate : {lr}, dropout: {dropout_value},l2: {regul_value} ')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 0.2)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    best_epoch = np.argmin(scikeras_regressor['estimator'].history_['val_loss']) + 1
    print('best epoch',best_epoch)
    best_epochs.append(best_epochs)
#%%
if model == 'ukesm':
    model_full = 'ukesm1-0-ll'
elif model == 'gfdl':
    model_full = 'gfdl-esm4'
elif model == 'ipsl':
    model_full = 'ipsl-cm6a-lr'
 
plt.figure(figsize=(8,6), dpi=300) #plot clusters
for model_full in ['gfdl-esm4', 'ipsl-cm6a-lr','ukesm1-0-ll']:
    for rcp_scenario in ['ssp126','ssp585']:
        DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_default_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)
        DS_y_epic_proj = convert_timeunit(DS_y_epic_proj)
        DS_y_epic_proj = DS_y_epic_proj.sel(time=slice(2015,2099))
        

        DS_y_epic_proj['yield-soy-noirr'].mean(['lat','lon']).plot(label = f'{model_full}_{rcp_scenario}')
        plt.legend()
plt.title('Global EPIC')        
plt.tight_layout()
plt.show()
 
    
plt.figure(figsize=(8,6), dpi=300) #plot clusters

for model_full in ['gfdl-esm4', 'ipsl-cm6a-lr','ukesm1-0-ll']:
    for rcp_scenario in ['ssp126','ssp585']:
        DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_default_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)
        DS_y_epic_proj = convert_timeunit(DS_y_epic_proj)
        DS_y_epic_proj = DS_y_epic_proj.sel(time=slice(2015,2099))
        DS_y_epic_proj_us = DS_y_epic_proj.where(DS_y_epic_us['yield'].mean('time') >= -5.0 )
        DS_y_epic_proj_br = DS_y_epic_proj.where(DS_y_epic_br['yield'].mean('time') >= -5.0 )
        DS_y_epic_proj_arg = DS_y_epic_proj.where(DS_y_epic_arg['yield'].mean('time') >= -5.0 )
        DS_y_epic_proj_br = DS_y_epic_proj_br.shift(time = 1) # SHIFT EPIC BR ONE YEAR FORWARD
        DS_y_epic_proj_arg = DS_y_epic_proj_arg.shift(time = 1) # SHIFT EPIC BR ONE YEAR FORWARD


        # Combine all grids
        DS_y_epic_proj_am = DS_y_epic_proj_us.combine_first(DS_y_epic_proj_br)
        DS_y_epic_proj_am = DS_y_epic_proj_am.combine_first(DS_y_epic_proj_arg)
        DS_y_epic_proj_am = DS_y_epic_proj_am.sel(time=slice(2017,2099))

        # Reindex to avoid missnig coordinates and dimension values
        DS_y_epic_proj_am = rearrange_latlot(DS_y_epic_proj_am)

        DS_y_epic_proj_am = DS_y_epic_proj_am.where(DS_y_obs_am_det.mean('time') > -10)
        DS_y_epic_proj_weight = weighted_conversion(DS_y_epic_proj_am, DS_area = DS_harvest_area_fut)

        DS_y_epic_proj_weight['yield-soy-noirr'].plot(label = f'{model_full}_{rcp_scenario}')
        plt.legend()
plt.title('EPIC RUNS')        
plt.tight_layout()
plt.show()

DS_y_epic_proj_am['yield-soy-noirr'].mean(['time']).plot()
plt.show()
DS_hybrid_trend_ukesm_85['yield-soy-noirr'].mean(['time']).plot()






#%%
for scenario in list(DS_hybrid_trend_all_weighted.keys()):
    DS_hybrid_trend_all_weighted[scenario].plot(label = scenario)
    DS_hybrid_all_weighted[scenario].plot(label = f'detrended {scenario}')
    plt.legend()
    plt.show()

DS_hybrid_gfdl_85_weighted['Yield'].plot(label = 'individual')
DS_hybrid_all_weighted['gfdl_85'].plot(label = 'all')
plt.legend()
plt.show()

DS_hybrid_ukesm_85_weighted['Yield'].plot(label = 'individual')
DS_hybrid_all_weighted['ukesm_85'].plot(label = 'all')
plt.legend()
plt.show()

DS_hybrid_ukesm_26_weighted['Yield'].plot(label = 'individual')
DS_hybrid_all_weighted['ukesm_26'].plot(label = 'all')
plt.legend()
plt.show()
