# eBayML_Functions.py
'''
Collection of functions used for feature extraction, predictions, and loss calcs
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn import linear_model
from sklearn.cluster import KMeans
import pgeocode
from tqdm import tqdm
import pickle

### Function to extract features from raw data set
def feature_extraction(df):
    # Create Feature Dataframe
    df_Feat =  pd.DataFrame(np.nan, index=df.index, columns=['Type','Handle','Ship_Method','Ship_Fee','Min','Max','Range','Item_Zip','Buyer_Zip',\
        'Dist','Weight','Category','Price','Quantity','Size','Processing_Days','Delivery_Days'])

    # Transaction Type (Bussiness to Consumer = 1, C2C = 0)
    df_Feat['Type'] = (df['b2c_c2c'] == 'B2C').astype(float)

    # Handling Days, clip at 4
    df_Feat['Handle'] = df['declared_handling_days']
    df_Feat['Handle'][df_Feat['Handle'] > 4] = 4

    # Shipment Method (1:15)
    df_Feat['Ship_Method'] = df['shipment_method_id']

    # Shipment Fee, clip at 5
    df_Feat['Ship_Fee'] = df['shipping_fee']
    df_Feat['Ship_Fee'][df_Feat['Ship_Fee'] > 5] = 5

    # Min/Max Estimates, Range of Estimate
    df_Feat['Min'] = abs(df['carrier_min_estimate'])
    df_Feat['Max'] = abs(df['carrier_max_estimate'])
    df_Feat['Range'] = abs(df['carrier_max_estimate'])- abs(df['carrier_min_estimate'])

    #Zip Codes
    df_Feat['Item_Zip'] = df['item_zip'].str[0:5].apply(pd.to_numeric, errors = 'coerce')
    df_Feat['Buyer_Zip'] = df['buyer_zip'].str[0:5].apply(pd.to_numeric, errors = 'coerce')

    # Get Postal Code Distance['Weight']
    postal_dist = pgeocode.GeoDistance('us')
    df_Feat['Dist'] = pd.Series(postal_dist.query_postal_code(df['item_zip'].values, df['buyer_zip'].values),index=df.index)

    # Weight, clip at 20lbs
    df_Feat['Weight'] = df['weight']
    df_Feat.loc[df['weight_units'] == 2]['Weight'] = (df[df['weight_units'] == 2]['weight'] * 2.2)
    df_Feat['Weight'][df_Feat['Weight'] > 20] = 20

    # Category ID
    df_Feat['Category'] = df['category_id']

    # Price, clip at 200
    df_Feat['Price'] = df['item_price']
    df_Feat['Price'][df_Feat['Price'] > 200] = 200

    # Quantity, clip at 2
    df_Feat['Quantity'] = df['quantity']
    df_Feat['Quantity'][df_Feat['Quantity'] > 2] = 2

    # Package Size (letter: 1, large env: 2, Thick Env: 3, large: 4, none: 0)
    df_Feat['Size'] = (df['package_size'] == 'LETTER') * 1 + (df['package_size'] == 'LARGE_ENVELOPE') * 2 + (df['package_size'] == 'PACKAGE_THICK_ENVELOPE') * 3 \
        + (df['package_size'] == 'LARGE_PACKAGE') * 4

    # Get Delivery Days to train on 
    if not(df.delivery_date.isna().all()):
        print('Extracting Training Features')
        for iRow in tqdm(df.index):
            payment = datetime.strptime(df.loc[iRow]['payment_datetime'][:16], '%Y-%m-%d %H:%M') + timedelta(hours = -int(df.loc[iRow]['payment_datetime'][-6:-3]))
            acceptance = datetime.strptime(df.loc[iRow]['acceptance_scan_timestamp'][:16], '%Y-%m-%d %H:%M') + timedelta(hours = -int(df.loc[iRow]['acceptance_scan_timestamp'][-6:-3]))
            delivery = datetime.strptime(df.loc[iRow]['delivery_date'],'%Y-%m-%d')
            processing = acceptance - payment
            total = delivery - payment

            df_Feat.at[iRow,'Processing_Days'] = processing.days
            df_Feat.at[iRow,'Delivery_Days'] = total.days

    else:
        print('Extracting Quiz/Test Features')
        for iRow in tqdm(df.index):
            payment = datetime.strptime(df.loc[iRow]['payment_datetime'][:16], '%Y-%m-%d %H:%M') + timedelta(hours = -int(df.loc[iRow]['payment_datetime'][-6:-3]))
            acceptance = datetime.strptime(df.loc[iRow]['acceptance_scan_timestamp'][:16], '%Y-%m-%d %H:%M') + timedelta(hours = -int(df.loc[iRow]['acceptance_scan_timestamp'][-6:-3]))
            processing = acceptance-payment

            df_Feat.at[iRow,'Processing_Days'] = processing.days

    return df_Feat

### Model Training

def train_kmeans(df,num_clusters,feature_list,all_feat=False,save_filename='Kmeans_model_noname',verbose=True):
    km = KMeans(n_clusters=num_clusters, init='random',n_init=10, max_iter=300,tol=1e-04, random_state=0)
    if all_feat: y_km = km.fit_predict(df.loc[:, df.columns != 'Delivery_Days'].values)
    else: y_km = km.fit_predict(df[feature_list].values)
    values, counts = np.unique(y_km, return_counts=True)
    if verbose: print('Done with K-Means {} Clusters\nCounts/Counts: {}'.format(num_clusters,counts))
    pickle.dump(km, open(save_filename, 'wb'))
    return km, y_km

def train_linears(df,num_clusters,cluster_assignments,feature_list,all_feat=False,save_filename='Linears_model_noname',verbose=True):
    linear_models = list()

    for iClust in tqdm(range(num_clusters), disable=not(verbose)):
        df_Clust = df[cluster_assignments == iClust]
        if all_feat: X = df_Clust.loc[:, df_Clust.columns != 'Delivery_Days'].values
        else: X = df_Clust[feature_list].values
        y = df_Clust['Delivery_Days'].values
        linear_models.append(linear_model.LinearRegression(normalize=True))
        linear_models[iClust].fit(X,y) 
        pickle.dump(linear_models[iClust], open(save_filename + '_' + str(iClust+1) + '.sav', 'wb'))

    return linear_models

### Loss Calculation

def calc_loss(df,km_model,lin_models,km_features,lin_features,all_feat=False,verbose=True):

    if all_feat: km_predict = km_model.predict(df.loc[:, df.columns != 'Delivery_Days'].values)
    else: km_predict = km_model.predict(df[km_features].values)

    predictions = np.empty(df.shape[0])

    for iClust in range(len(lin_models)):
        df_Clust = df[km_predict == iClust]
        if all_feat: X = df_Clust.loc[:, df_Clust.columns != 'Delivery_Days'].values
        else: X = df_Clust[lin_features].values
        predictions[km_predict == iClust] = lin_models[iClust].predict(X).astype(int)

    given_days = df['Delivery_Days'].values
    error_days = given_days - predictions
    early = 0.4 * abs(np.multiply(error_days < 0, error_days).sum())
    late = 0.6 * np.multiply(error_days > 0, error_days).sum()
    loss = (early + late)/df.shape[0]
    if verbose: print('Loss is {:.2f}'.format(loss))
    return loss, predictions 

### Quiz Data

def predict(df,km_model,lin_models,km_features,lin_features,all_feat=False,verbose=True):

    if all_feat: km_predict = km_model.predict(df.loc[:, df.columns != 'Delivery_Days'].values)
    else: km_predict = km_model.predict(df[km_features].values)

    predictions = np.empty(df.shape[0])

    for iClust in range(len(lin_models)):
        df_Clust = df[km_predict == iClust]
        if all_feat: X = df_Clust.loc[:, df_Clust.columns != 'Delivery_Days'].values
        else: X = df_Clust[lin_features].values
        predictions[km_predict == iClust] = lin_models[iClust].predict(X).astype(int)

    return predictions 

def saveQuiz(df,predictions,filename):
    df_out =  pd.DataFrame(np.nan, index=df.index, columns=['record identifier','predicted delivery date']).astype(str)

    for iRow in tqdm(range(df.shape[0])):
        payment = datetime.strptime(df.iloc[iRow]['payment_datetime'][:16], '%Y-%m-%d %H:%M') + timedelta(hours = -int(df_quiz.iloc[iRow]['payment_datetime'][-6:-3]))
        delivery = payment + timedelta(days=predictions[iRow])
        df_out.at[iRow,'record identifier'] = df_quiz['record_number'][iRow]
        df_out.at[iRow,'predicted delivery date'] = delivery.strftime('%Y-%m-%d')

    df_out.to_csv(filename, sep="\t",header=False,index=False, compression= 'gzip')
