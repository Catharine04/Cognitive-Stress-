# views.py
from django.http import JsonResponse
import pandas as pd
from django.http import JsonResponse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from django.http import JsonResponse
from io import StringIO


def process_csv(request):
    if request.method == 'POST':
        
        csv_data = request.body.decode('utf-8') 
        if csv_data:
            # Process the CSV data here
           
            print("this is csv data", csv_data)
            
        
            track_ids = generate_music_recommendations(csv_data)
            return JsonResponse({'track_ids': track_ids})
        else:
            return JsonResponse({'error': 'No CSV file provided'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


def generate_music_recommendations(csv_data):
    try:
        # Load dataset
        print("hihihi")
        dataset = pd.read_csv('dataset.csv')
        
        # Sample 150 songs from each genre
        grouped = dataset.groupby('track_genre')
        df = pd.DataFrame()
        for genre, group in grouped:
            sampled_group = group.sample(n=150, random_state=42)
            df = pd.concat([df, sampled_group])
        df.reset_index(drop=True, inplace=True)
        print(df)
        
        # Drop unnecessary columns
        df.drop(columns=['Unnamed: 0', 'artists', 'album_name', 'track_name', 'popularity', 'explicit', 'track_genre','time_signature', 'duration_ms'], axis=1, inplace=True)
        
        # Standardize the features
        original_dataset=df.copy()
        print("this is original dataset",original_dataset)
       
        
        # Remove unnecessary features
        df.drop(columns=['track_id'], axis=1, inplace=True)
        print("thsiis is df ",df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        print("sudhisdho",X_scaled)
        print("this is original dataset part 2",original_dataset)


    
        # Determine optimal number of clusters using silhouette score
        silhouette_scores = []
        k_range = range(5, 15)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 5
        print("this si silhoutee dcore",silhouette_scores)
        print("hiejlsdjo",optimal_clusters)
       
        # Apply KMeans clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        kmeans.fit(X_scaled)
        cluster_labels = kmeans.labels_
        print("hfeijdl",cluster_labels)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Initialize the KMeans model with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        print("this is knmenss",kmeans)

        # Fit the model to the reduced data
        kmeans.fit(X_pca)

        cluster_labels = kmeans.labels_
        df['Cluster'] = cluster_labels
        print("this is cluster label",df)

        cluster_centroids = df.groupby('Cluster').mean()
 
        cluster_centroids = cluster_centroids.fillna(cluster_centroids.mean())
        print("this is clyseter centroud",cluster_centroids)
    
        user_data = pd.read_csv(StringIO(csv_data))
        print("this is csv datra ",user_data)
        user_data.drop('id', axis=1, inplace=True)
        user_data = user_data.fillna(user_data.mean())
        nan_values = user_data.isna().sum()
        print("hhugygyfftdt")

        try:
    # # Calculate cosine similarity between user's songs' features and cluster centroids
            similarity_scores = cosine_similarity(user_data.values, cluster_centroids.values)
            print("vjhgkhkhgcvhj",similarity_scores)
        except Exception as error:
             print("An error occurred:", error)

    #     print("gchgfjk")
    #     print(similarity_scores)

        assigned_clusters = similarity_scores.argmax(axis=1)
        print("bdjksla",assigned_clusters)
        majority_cluster = pd.Series(assigned_clusters).value_counts().idxmax()
        print("sdhljdhbsk",majority_cluster)
        print("this is original datset",original_dataset)
        try:
            recommendations = df[df['Cluster'] == majority_cluster].sample(5)
        except Exception as error:
             print("An error occurred:", error)

        print(recommendations)
        recommendation_index = recommendations.index[0]
        print("htis is ",recommendation_index)
        recommendation_indices = []

        # Iterate over each index in the recommendation DataFrame
        for index in recommendations.index:
            # Append the index to the list
            recommendation_indices.append(index)

        # Print the list of recommendation indices
        print("Recommendation Indices:", recommendation_indices) 
        track_ids = []

        # Loop through each recommendation index
        for index in recommendation_indices:
            # Retrieve the row from the original dataset at the specified index
            row = original_dataset.iloc[index]
            # Get the track ID from the row and append it to the list
            track_id = row['track_id']
            track_ids.append(track_id)

        # Print the track IDs
        print("Track IDs:", track_ids)
    
        
        return track_ids
    except Exception as e:
        return {'error': str(e)}
