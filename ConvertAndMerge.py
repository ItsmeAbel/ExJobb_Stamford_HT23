import pandas as pd

#Load csv
df1 = pd.read_csv("CSV_FILES\exjobb_export_pricechange.csv", sep=';')
df2 = pd.read_csv("CSV_FILES\exjobb_export_item.csv", sep=';', encoding = 'ANSI')

#Print original tables
print("Numpers of rows in the original tables: ", df1.shape[0], " and ", df2.shape[0])

#Change seperation format
df1 = df1.replace('\.', '', regex=True)
df1 = df1.replace(',', '.', regex=True)

df2 = df2.replace('\.', '', regex=True)
df2 = df2.replace(',', '.', regex=True)

#Reload csv
df1.to_csv("CSV_FILES\TEMPORARY_CSV\complete_data.csv", index = False)
df1 = pd.read_csv("CSV_FILES\TEMPORARY_CSV\complete_data.csv")
df2.to_csv("CSV_FILES\TEMPORARY_CSV\complete_data.csv", index = False)
df2 = pd.read_csv("CSV_FILES\TEMPORARY_CSV\complete_data.csv")

merged_df = pd.merge(df1, df2, on='item_obj', how='left')

merged_df.to_csv('CSV_FILES\TEMPORARY_CSV\complete_data.csv', index = False)

print("Numpers of rows in the final table: ", merged_df.shape[0])