#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('dataset.csv')


# In[3]:


dataset.shape


# In[4]:


dataset.head()


# In[10]:


unique_genres = dataset['track_genre'].unique()


# In[11]:


unique_genres


# In[12]:


# Group the dataset by genre
grouped = dataset.groupby('track_genre')

# Create an empty DataFrame to store the sampled data
df = pd.DataFrame()

# Sample 200 songs from each genre group and concatenate them
for genre, group in grouped:
    sampled_group = group.sample(n=150, random_state=42)
    df = pd.concat([df, sampled_group])

# Reset the index of the sampled data
df.reset_index(drop=True, inplace=True)

# Display the first few rows of the sampled data
print(df.head())



# In[13]:


print(df.shape)  # Check the shape of the sampled data


# In[14]:


genre_counts = df.groupby('track_genre').size()
print(genre_counts)


# In[15]:


missing_values = df.isnull().sum()
print(missing_values)


# In[16]:


df.drop(columns=['Unnamed: 0','artists','album_name','track_name','popularity','explicit','track_genre','track_id'], axis=1,inplace= True)

df.head()


# In[17]:


df.head()


# In[18]:


df.corr()


# In[19]:


from sklearn.preprocessing import StandardScaler

# Assuming X is your dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[20]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Assuming X_scaled is your preprocessed data
wcss = []
silhouette_scores = []
k_range = range(5, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.show()


# In[21]:


from sklearn.metrics import silhouette_score

# Range of clusters to try
k_range = range(5, 15)  # You can adjust this range as needed

# List to store silhouette scores for each number of clusters
silhouette_scores = []

# Iterate over the range of clusters
for k in k_range:
    # Initialize the KMeans model with k clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Fit the model to the scaled data
    kmeans.fit(X_scaled)
    
    # Get cluster labels
    cluster_labels = kmeans.labels_
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # Append silhouette score to the list
    silhouette_scores.append(silhouette_avg)


# In[22]:


silhouette_scores


# In[23]:


##### from sklearn.cluster import KMeans

# Initialize the KMeans model with the desired number of clusters
kmeans = KMeans(n_clusters=16, random_state=42)

# Fit the model to the scaled data
kmeans.fit(X_scaled)

# Get cluster labels
cluster_labels = kmeans.labels_

# Print the cluster labels for each data point
print(cluster_labels)


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

# Reduce dimensionality of data (if needed) using PCA or t-SNE
# For example, using PCA:
from sklearn.decomposition import PCA
pca = PCA(n_components=2 )
X_pca = pca.fit_transform(X_scaled)  # X_scaled is your scaled data

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', legend='full')
plt.title('Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[26]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize RFE with the Random Forest classifier
rfe = RFE(estimator=rf_classifier, n_features_to_select=10)  # Choose the number of features to select

# Fit RFE to your data
rfe.fit(X_scaled, cluster_labels)  # X_scaled is your scaled dataset, cluster_labels are the cluster labels

# Get selected feature indices
selected_feature_indices = rfe.get_support(indices=True)

# Get the names of selected features
selected_features = df.columns[selected_feature_indices]

print("Selected Features:")
print(selected_features)


# In[27]:


df.drop(columns=['time_signature','duration_ms'], axis=1,inplace= True)
df


# In[28]:


from sklearn.preprocessing import StandardScaler

# Assuming X is your dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[29]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Assuming X_scaled is your preprocessed data
wcss = []
silhouette_scores = []
k_range = range(5, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.show()


# In[30]:


from sklearn.metrics import silhouette_score

# Range of clusters to try
k_range = range(5, 15)  # You can adjust this range as needed

# List to store silhouette scores for each number of clusters
silhouette_scores = []

# Iterate over the range of clusters
for k in k_range:
    # Initialize the KMeans model with k clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Fit the model to the scaled data
    kmeans.fit(X_scaled)
    
    # Get cluster labels
    cluster_labels = kmeans.labels_
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # Append silhouette score to the list
    silhouette_scores.append(silhouette_avg)


# In[31]:


silhouette_scores


# In[32]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Silhouette scores
silhouette_scores = [0.12637417645276094, 0.13622961109575715, 0.12020579964153726,
                     0.1369705884764036, 0.14249084763496658, 0.1399963181444394,
                     0.13440762269680978, 0.1350442574796413, 0.13375108857029602,
                     0.12772208604252236]

# Find the index of the maximum silhouette score
optimal_index = silhouette_scores.index(max(silhouette_scores))

# Optimal number of clusters
optimal_clusters = optimal_index + 5  # Adding 5 because the range starts from 5

# Initialize the KMeans model with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

# Fit the model to the scaled data
kmeans.fit(X_scaled)

# Get cluster labels
cluster_labels = kmeans.labels_

# Print the optimal number of clusters
print("Optimal number of clusters:", optimal_clusters)

# Print the cluster labels
print("Cluster labels:", cluster_labels)


# In[33]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensionality to 2 components using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Initialize the KMeans model with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

# Fit the model to the reduced data
kmeans.fit(X_pca)

# Get cluster labels
cluster_labels = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
plt.title('KMeans Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()


# In[1]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load user data
user_data = pd.read_csv('music_features.csv')
print(user_data)

# Drop the 'id' column if present
user_data = user_data.drop('id', axis=1)

# Load clustered music data
clustered_music_data = pd.read_csv('clustered_music_data.csv')


# Calculate cosine similarity between user's songs' features and cluster centroids
cluster_centroids = clustered_music_data.groupby('Cluster').mean()
cluster_centroids.reset_index(inplace=True)



# # Calculate cosine similarity between user's songs and cluster centroids
# similarity_scores = cosine_similarity(user_data.values, cluster_centroids.values)

# # Assign user's songs to cluster(s) with highest similarity scores
# assigned_clusters = similarity_scores.argmax(axis=1)

# # Find the cluster with the majority of assigned songs
# majority_cluster = pd.Series(assigned_clusters).value_counts().idxmax()

# # Get songs from the majority cluster as recommendations
# recommendations = clustered_music_data[clustered_music_data['Cluster'] == majority_cluster].sample(5)

# # Display recommendations to the user
# print("Recommended Songs:")
# print(recommendations)

clustered_music_data.shape



# In[35]:


cluster_centroids.shape


# In[36]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load user data
user_data = pd.read_csv('music_features.csv')

# Drop the 'id' column if present
user_data = user_data.drop('id', axis=1)

# Load clustered music data
clustered_music_data = pd.read_csv('clustered_music_data.csv')

# Extract features from user's listened songs data

# Calculate cosine similarity between user's songs' features and cluster centroids
cluster_centroids = clustered_music_data.groupby('Cluster').mean()

cluster_centroids.values
cluster_centroids.reset_index(inplace=True)
cluster_centroids

similarity_scores = cosine_similarity(clustered_music_data.values, cluster_centroids.values)

# Assign user to cluster(s) with highest similarity scores
assigned_clusters = similarity_scores.argmax(axis=1)
print("this is assigned clusters",assigned_clusters)

# Get songs from assigned cluster(s) as recommendations
recommendations = clustered_music_data[clustered_music_data['Cluster'].isin(assigned_clusters)]

# Display recommendations to the user
print("Recommended Songs:")
recommendations.shape
# print(recommendations)
# print("Clustered Music Data Shape:", clustered_music_data.values.shape) 
# print("Cluster Centroids Shape:", cluster_centroids.values.shape)


# In[44]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load user data
user_data = pd.read_csv('music_features.csv')

# Drop the 'id' column if present
user_data = user_data.drop('id', axis=1)


# Fill NaN values with the mean of each column
user_data = user_data.fillna(user_data.mean())
nan_values = user_data.isna().sum()


# Load clustered music data
clustered_music_data = pd.read_csv('clustered_music_data.csv')
nan_valuessss = clustered_music_data.isna().sum()

# # Calculate cosine similarity between user's songs' features and cluster centroids
cluster_centroids = clustered_music_data.groupby('Cluster').mean()

# # Fill NaN values with the mean of each column
cluster_centroids = cluster_centroids.fillna(cluster_centroids.mean())

# # Calculate cosine similarity between user's songs and cluster centroids
similarity_scores = cosine_similarity(user_data.values, cluster_centroids.values)

# # Assign user's songs to cluster(s) with highest similarity scores
assigned_clusters = similarity_scores.argmax(axis=1)

# Find the cluster with the majority of assigned songs
majority_cluster = pd.Series(assigned_clusters).value_counts().idxmax()

# Get songs from the majority cluster as recommendations
recommendations = clustered_music_data[clustered_music_data['Cluster'] == majority_cluster].sample(5)

# Display recommendations to the user
print("Recommended Songs:")
print(recommendations)


# In[3]:


recommendations


# In[ ]:




