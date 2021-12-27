# eBayML_Functions.py
'''
Collection of functions used for feature extraction, predictions, and loss calcs
'''

import pandas as pd
import pgeocode
from tqdm import tqdm


##### Data Formatting #######
def feature_extraction(df):
    # Create Feature Dataframe
    df_Feat =  pd.DataFrame(np.nan, index=df.index, columns=['Type','Handle','Ship_Method','Min','Max','Range','Dist','Weight',\
        'Category','Price','Quantity','Size','Delivery_Days'])

    # Transaction Type (Bussiness to Consumer = 1.5, C2C = .5)
    df_Feat['Type'] = (df['b2c_c2c'] == 'B2C').astype(float) + 0.5

    # Handling Days
    df_Feat['Handle'] = df['declared_handling_days']

    # SHipment Method (1:15)
    df_Feat['Ship_Method'] = df['shipment_method_id'] + 1

    # Min/Max Estimates, Range of Estimate
    df_Feat['Min'] = abs(df['carrier_min_estimate'])
    df_Feat['Max'] = abs(df['carrier_max_estimate'])
    df_Feat['Range'] = abs(df['carrier_max_estimate'])- abs(df['carrier_min_estimate'])

    # Get Postal Code Distance['Weight']
    postal_dist = pgeocode.GeoDistance('us')
    df_Feat['Dist'] = pd.Series(postal_dist.query_postal_code(df['item_zip'].values, df['buyer_zip'].values),index=df.index)

    # Weight
    df_Feat['Weight'] = df['weight']
    df_Feat.loc[df['weight_units'] == 2]['Weight'] = (df[df['weight_units'] == 2]['weight'] * 2.2)

    # Category ID (shift up one)
    df_Feat['Category'] = df['category_id'] + 1

    # Price
    df_Feat['Price'] = df['item_price']

    # Quantity
    df_Feat['Quantity'] = df['quantity']

    # Package Size (letter: 1, large env: 2, Thick Env: 3, large: 4, none: 0)
    df_Feat['Size'] = (df['package_size'] == 'LETTER') * 1 + (df['package_size'] == 'LARGE_ENVELOPE') * 2 + (df['package_size'] == 'PACKAGE_THICK_ENVELOPE') * 3 \
        + (df['package_size'] == 'LARGE_PACKAGE') * 4

    # Get Delivery Days to train on 
    if not(df.delivery_date.isna().all()):
        for iRow in tqdm(range(df.shape[0])):
            payment = datetime.strptime(df.iloc[iRow]['payment_datetime'][:16], '%Y-%m-%d %H:%M') + timedelta(hours = -int(df.iloc[iRow]['payment_datetime'][-6:-3]))
            delivery = datetime.strptime(df.iloc[iRow]['delivery_date'],'%Y-%m-%d')
            difference = delivery - payment
            df_Feat.at[iRow,'Delivery_Days'] = difference.days

    return df_Feat

def calc_loss(model,df):
    X = df.values[:,0:-1]
    predictions = model.predict(X).astype(int)
    truth = df.values[:,-1]
    difference = truth - predictions
    early = 0.4 * abs(np.multiply(difference < 0, difference).sum())
    late = 0.6 * np.multiply(difference > 0, difference).sum()
    loss = (early + late)/df.shape[0]
    print('Loss is {:.2f}'.format(loss))
    return loss

def calc_loss_clust(models,df):

    km_predict = km.predict(df.values)
    predictions = np.empty(df.shape[0])

    for iClust in range(len(models)):
        df_Clust = df[km_predict == iClust]
        X = df_Clust.values[:,0:-1]
        predictions[km_predict == iClust] = models[iClust].predict(X).astype(int)

    truth = df.values[:,-1]
    difference = truth - predictions
    early = 0.4 * abs(np.multiply(difference < 0, difference).sum())
    late = 0.6 * np.multiply(difference > 0, difference).sum()
    loss = (early + late)/df.shape[0]
    print('Loss is {:.2f}'.format(loss))
    return loss, predictions 

def predict(models,df):

    km_predict = km.predict(df.values)
    predictions = np.empty(df.shape[0])

    for iClust in range(len(models)):
        df_Clust = df[km_predict == iClust]
        X = df_Clust.values[:,0:-1]
        predictions[km_predict == iClust] = models[iClust].predict(X).astype(int)

    return predictions 
    