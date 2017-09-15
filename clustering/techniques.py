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

#nice dendrogram
# plot dendrogram
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

#use dendrogram to deterimne how many clusters to make
#use dendrogram to deterimne how many clusters to make
max_d = 1700
fancy_dendrogram(
           Z, 
           truncate_mode='lastp',       # show last p merged clusters
           p=12,                        # show last p merged clusters
           leaf_rotation=0.,            # rotate text labels
           leaf_font_size=15,           # label size
           show_contracted=True,         # show height of merged clusters as dots
           annotate_above=200,
           max_d = max_d
       )

#hierarchical elbow graph
last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print ("clusters:", k)

