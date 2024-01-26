import pandas as pd

REGRESSION_LIMIT_TERM = 0.01

#Load csv
df = pd.read_csv("CSV_FILES\TEMPORARY_CSV\complete_data.csv")

#Print original table
print("Numpers of rows in the original table: ", df.shape[0])
print("Numpers of colums in the original table: ", df.shape[1])

#Set columntypes
df['old_cost_price'] = df['old_cost_price'].astype(float)
df['new_cost_price'] = df['new_cost_price'].astype(float)
df['old_agreement_price'] = df['old_agreement_price'].astype(float)
df['new_agreement_price'] = df['new_agreement_price'].astype(float)

#Calculate new agreement price estimation
def ap_estimate(row):
    ocp = row['old_cost_price']
    ncp = row['new_cost_price']
    oap = row['old_agreement_price']
    ap_est = ncp/ocp * oap if ocp > REGRESSION_LIMIT_TERM else -1

    return ap_est
df['new_ap_est'] = df.apply(ap_estimate, axis=1)

#Change date format
df['change_date'] = pd.to_datetime(df['change_date']).astype('int64') // 10**9


#Remove NULL values
condition1 = df['old_cost_price'] == 0
condition2 = df['new_cost_price'] == 0
condition3 = df['old_agreement_price'] == 0
condition4 = df['new_agreement_price'] == 0
condition0 = df['new_ap_est'] == -1
condition = condition1 | condition2 | condition3 | condition4 | condition0

df = df[~condition]
df = df.reset_index(drop=True)

#Pick desired columns
new_df = df[['old_cost_price',
             'new_cost_price',
             'old_agreement_price',
             'new_agreement_price',
             'new_ap_est', 
             'change_date']]

category_dummies = pd.get_dummies(df['item_category_1'], prefix=['item_category_1']).astype(int)

new_df = pd.concat([new_df, category_dummies], axis=1)

cutoff = int(0.99 * len(new_df))

#Load 99% of input data to csv file
new_df[:cutoff].to_csv("priceChange.csv", index=False) 

#Load 1% to new csv file. Unseen da
new_df[cutoff:].to_csv("priceChange2.csv", index=False)

#Print updated table
print("Numpers of rows in the new table: ", new_df.shape[0])
print("Numpers of colums in the original table: ", df.shape[1])