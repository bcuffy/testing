#techniques

# replace column name pandas
df = pd.read_csv('auto-mpg2.csv')
df = df.rename(columns={df.columns[0]: 'name'})
df.drop(['name'], 1, inplace=True)

# Remove header row
    # header = df.copy(deep=True).columns
    # df.columns = df.iloc[0]
    # df =df[0:]
    # print(list(header))



# Remove header row
df.columns = df.iloc[0]

