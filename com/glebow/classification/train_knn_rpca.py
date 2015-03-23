__author__ = 'pglebow'
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

pca = RandomizedPCA(n_components=5)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

print train_x[:5]
#array([[ 12614.55016475,  -9156.62662224,  -7649.37090539,  -3230.94749506,
#          2495.71170459],
#       [ 16111.39363837,   -259.55063579,    699.60464599,   3058.59026495,
#         -1552.34714653],
#       [ 15019.71069584,  -6403.86621428,   1968.44401114,   2896.76676466,
#         -2157.76499726],
#       [ 13410.53053415,  -1658.3751377 ,    261.26829049,   1991.33404567,
#          -486.60683822],
#       [ 12717.28773107,  -1544.27233216,  -1279.70167969,    503.33658729,
#           -38.00244617]])

knn = KNeighborsClassifier()
knn.fit(train_x, train_y)