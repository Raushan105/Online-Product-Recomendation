import sqlite3
import pandas as pd
import numpy as np
#!pip install mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


adhesiveSealantTrans = pd.read_csv("/Users/DJ/E-Commerce-Recommendations/e-comm-data/Transactions with A&S.txt",sep='\t', header=0)
allTransaction = pd.read_csv("/Users/DJ/E-Commerce-Recommendations/e-comm-data/All Transations - 2 Weeks.txt", sep='\t', header=0)

adhesiveSealantTrans.head(5)

allTransaction.head(5)

frames = [adhesiveSealantTrans, allTransaction]
allTransactions = pd.concat(frames)

# Check data loaded correctly
allTransactions.head()

allTransactions.shape

print("All Transactions Data", allTransactions.dtypes)


# database
db = sqlite3.connect("transactions.db")

# Connecting to database
con = sqlite3.connect("/Users/DJ/E-Commerce-Recommendations/transactions.db")
# Move dataframe to sql table in transactions.db
allTransactions.to_sql("all_trans", con, if_exists="replace", index=False)
# Unique values for all features
print("Unique Order Numbers: \n", pd.read_sql_query("SELECT COUNT(DISTINCT order_number) FROM all_Trans;", con))
print("\nUnique Brands: \n", pd.read_sql_query("SELECT COUNT(DISTINCT Brand) FROM all_Trans;", con))
print("\nUnique Sku: \n", pd.read_sql_query("SELECT COUNT(DISTINCT Sku) FROM all_Trans;", con))
print("\nUnique l1: \n", pd.read_sql_query("SELECT COUNT(DISTINCT l1) FROM all_Trans;", con))
print("\nUnique l2: \n", pd.read_sql_query("SELECT COUNT(DISTINCT l2) FROM all_Trans;", con))
print("\nUnique l3: \n", pd.read_sql_query("SELECT COUNT(DISTINCT l3) FROM all_Trans;", con))


# Missing data
print("\nAll Transactions Missing Data")
print("Brand", pd.read_sql_query("SELECT CAST(SUM(CASE WHEN brand is NULL THEN 1 ELSE 0 END) as float)/COUNT(*) as ProportionMissing FROM all_trans;", con))
print("Sku", pd.read_sql_query("SELECT CAST(SUM(CASE WHEN sku is NULL THEN 1 ELSE 0 END) as float)/COUNT(*) as ProportionMissing FROM all_trans;", con))
print("L1", pd.read_sql_query("SELECT CAST(SUM(CASE WHEN l1 is NULL THEN 1 ELSE 0 END) as float)/COUNT(*) as ProportionMissing FROM all_trans;", con))
print("L2", pd.read_sql_query("SELECT CAST(SUM(CASE WHEN l2 is NULL THEN 1 ELSE 0 END) as float)/COUNT(*) as ProportionMissing FROM all_trans;", con))
print("L3", pd.read_sql_query("SELECT CAST(SUM(CASE WHEN l3 is NULL THEN 1 ELSE 0 END) as float)/COUNT(*) as ProportionMissing FROM all_trans;", con))


# ALL transactons
pd.read_sql_query("SELECT l1,l2, l3, COUNT(sku) FROM all_trans GROUP BY sku ORDER BY COUNT(sku) DESC LIMIT 10;", con)



allTransactions.drop('sku', axis=1, inplace=True)
allTransactions.drop('brand', axis=1, inplace=True)

allTransactions.drop('l1', axis=1, inplace=True)
allTransactions.drop('l2', axis=1, inplace=True)

grouped = allTransactions.groupby('order_number')['l3'].apply(list)
grouped.head()

 
grouped.shape

filtered_group = [x for x in grouped if len(x)>=10 ]
filtered_group[:3]

# One-hot encode data in pandas dataframe
te = TransactionEncoder()
te_ary = te.fit(filtered_group).transform(filtered_group)
transaction_group = pd.DataFrame(te_ary, columns=te.columns_)
transaction_group.head()

apriori(transaction_group, min_support=0.01, use_colnames=True)


freq_itemsets = apriori(transaction_group, min_support=0.01, use_colnames=True)
freq_itemsets['length'] = freq_itemsets['itemsets'].apply(lambda x : len(x))
freq_itemsets.head()


#Filter out single-item orders 
freq_itemsets[ (freq_itemsets['length'] > 1) &
             (freq_itemsets['support'] > 0.02) ]
             
             
rules = association_rules(freq_itemsets, metric="lift", min_threshold=1)
rules.head()


# Filter items and save
saved_recs = rules[ (rules['lift'] >= 6) &
     rules['confidence'] >= 0.8]

# Check saved dataframe
saved_recs.sort_values(by=['confidence'], ascending=False)


# Save Recs to csv
saved_recs.to_csv('recommendations.csv', index=False)
