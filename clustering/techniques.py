#techniques

# replace column name pandas
df = pd.read_csv('auto-mpg2.csv')
df = df.rename(columns={df.columns[0]: 'name'})
df.drop(['name'], 1, inplace=True)

# Remove header row
df.columns = df.iloc[0]