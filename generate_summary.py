import pandas as pd
df = pd.read_excel("ABC_Consumer_Products_Sales_Dataset_FY2025-26.xlsx", sheet_name="Transactions")
print(df.shape)