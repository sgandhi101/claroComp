import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the data into a pandas dataframe
df = pd.read_csv('inpatient_discharges_first_six.csv')

df['Expected Mortality'] = df['Expected Mortality'].str.strip('%')
# convert to float
df['Expected Mortality'] = df['Expected Mortality'].astype(float)
# divide by 100 to get the actual value
df['Expected Mortality'] = df['Expected Mortality'] / 100

# Create a dictionary to map each categorical value to a numerical value
class_map = {'MCC': 1, 'CC': 2, 'NCC': 3}

# Use the map function to convert the 'Class' column
df['Class'] = df['Class'].map(class_map)

unique_values = df['Pridx'].drop_duplicates().tolist()
# Map each unique value to an integer
value_map = {value: index + 1 for index, value in enumerate(unique_values)}

# Replace the values in the 'Pridx' column with the corresponding integers
df['Pridx'] = df['Pridx'].map(value_map)

unique_values = df['DRG Description'].drop_duplicates().tolist()
# Map each unique value to an integer
value_map = {value: index + 1 for index, value in enumerate(unique_values)}

# Replace the values in the 'Pridx' column with the corresponding integers
df['DRG Description'] = df['DRG Description'].map(value_map)

unique_values = df['Secdx1'].drop_duplicates().tolist()
# Map each unique value to an integer
value_map = {value: index + 1 for index, value in enumerate(unique_values)}

# Replace the values in the 'Pridx' column with the corresponding integers
df['Secdx1'] = df['Secdx1'].map(value_map)

unique_values = df['M/S'].drop_duplicates().tolist()
# Map each unique value to an integer
value_map = {value: index + 1 for index, value in enumerate(unique_values)}

# Replace the values in the 'Pridx' column with the corresponding integers
df['M/S'] = df['M/S'].map(value_map)

# Select the columns to use for clustering
columns_to_use = ['DRG Weight', 'Expected Mortality', 'GMLOS', 'DRG Description', 'Pridx', 'Dishcarge Status', 'Secdx1']
df = df[columns_to_use]

# Standardize the data to normalize the scales of the columns
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Find the optimal number of clusters
silhouette_scores = []
lowest = 5
highest = 50
for n_clusters in range(lowest, highest):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df_scaled)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot the silhouette scores to visualize the results
plt.plot(range(lowest, highest), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Find the number of clusters with the highest silhouette score
optimal_n_clusters = np.argmax(silhouette_scores) + 2
print('The optimal number of clusters is:', optimal_n_clusters)

# Apply KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters)
kmeans.fit(df_scaled)
df['cluster'] = kmeans.labels_

# Apply KMeans clustering to the scaled data
kmeans = KMeans(n_clusters=optimal_n_clusters)
kmeans.fit(df_scaled)

# Add the cluster labels to the original dataframe
df['cluster'] = kmeans.labels_

# Analyze the results of the clustering
print(df.groupby(['cluster']).mean())

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the scatterplot of the data colored by the cluster label
sns.scatterplot(x='DRG Weight', y='Expected Mortality', hue='cluster', data=df)
plt.show()

# Plot the scatterplot of the data colored by the cluster label
sns.scatterplot(x='GMLOS', y='Expected Mortality', hue='cluster', data=df)
plt.show()

# Plot the scatterplot of the data colored by the cluster label
sns.scatterplot(x='DRG Description', y='Expected Mortality', hue='cluster', data=df)
plt.show()

# Plot the scatterplot of the data colored by the cluster label
sns.scatterplot(x='Pridx', y='Expected Mortality', hue='cluster', data=df)
plt.show()

# Plot the scatterplot of the data colored by the cluster label
sns.scatterplot(x='Dishcarge Status', y='Expected Mortality', hue='cluster', data=df)
plt.show()


# Plot the scatterplot of the data colored by the cluster label
sns.scatterplot(x='Secdx1', y='Expected Mortality', hue='cluster', data=df)
plt.show()

print('The optimal number of clusters is:', optimal_n_clusters)
print(df.groupby(['cluster']).mean())
