import sys
from create_clusters import *
from cluster_prediction import *
import matplotlib.pyplot as plt

K_clusters = 20
min_length = 20
n_dimensions = 30
errors = []
accuracy = []


X = dim_reduction_svd(clean_data(min_length), n_dimensions)
print(X.shape)
Y, rss = create_kmeans_clusters(X, K_clusters)
print("RSS = " + str(rss))
accuracy = (predict_NB_Bernoulli(X, Y))
print("Accuracy = " + str(accuracy))
