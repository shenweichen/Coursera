import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('./places.txt',names=['lon','lat'])
clf = KMeans(n_clusters=3,n_jobs=-1)
data['place'] = clf.fit_predict(data)
data.to_csv('clusters.txt',sep=' ',columns=['place'],header=False)