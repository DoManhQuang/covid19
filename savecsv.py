import pandas as pd

data = [[1, 2]]

df = pd.DataFrame(data, columns=['Product', 'Price'])

df.to_csv ('dataframe.csv', index=True, header=True)

print (df)