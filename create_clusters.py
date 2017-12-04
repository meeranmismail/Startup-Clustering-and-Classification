import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
#################### Full Data #########################

np.random.seed(42)
stop = stopwords.words('english')

def clean_data(length_mask = 0):
	"""Return a list of companies that have descriptions from both pitchbook and crunchbase, with a length of at least length_mask (defaults to length_mask = 0)."""
	training_categories_df = pd.read_csv("/users/Meeranster/CS341/data/raw_data_fixed.csv",  encoding = "ISO-8859-1", usecols=['domain'\
	, 'tx_industry', 'cb_category', 'tx_category', 'cb_desc', 'pb_desc', 'pb_category'])

	training_categories_df["desc"] = training_categories_df["pb_desc"] + training_categories_df["cb_desc"]
	subset_df_desc = training_categories_df[(training_categories_df['desc'].notnull())]
	subset_df_desc = subset_df_desc[subset_df_desc["desc"].str.split().str.len() > length_mask]
	len_of_df = subset_df_desc.shape[0]
	print("There are " + str(len_of_df) + " companies that have a full provided description out of " + str(training_categories_df.shape[0]) + " companies.")
	subset_df_desc.sample(frac=1, random_state = 42).reset_index(drop=True)
	subset_df_desc['desc'] = subset_df_desc['desc'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	X = get_feature_matrix(subset_df_desc)
	sparsity = X.sum()/float(X.shape[0]*X.shape[1])
	print("Term-document sparsity: " + str(sparsity))
	return X

#use if you want to get a sparse np matrix that can be used by scikit learn
def get_feature_matrix(startup_list, is_indicator = True):
	"""Return a feature matrix for each word in the array."""
	countvec = CountVectorizer(binary = is_indicator)
	return countvec.fit_transform(startup_list['desc'])



def create_random_clusters(K_clusters, m_examples):
	return np.random.randint(0, K_clusters, m_examples)

def dim_reduction_svd(X, n_dimensions):
	svd = TruncatedSVD(n_components=n_dimensions, random_state=42)
	svd.fit(X)
	return svd.transform(X)


def create_kmeans_clusters(X, K_clusters):
	k_means = KMeans(n_clusters=K_clusters, random_state=42).fit(X)
	print("Average reconstruction loss: " + str(k_means.score(X, k_means.labels_)/X.shape[1]))
	return k_means.labels_, k_means.score(X, k_means.labels_)/X.shape[1]
















