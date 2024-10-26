# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation.
2.Choosing the Number of Clusters (K).
3.K-Means Algorithm Implementation.
4.Evaluate Clustering Results.
5.Deploy and Monitor. 


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:PREM PRASANTH.J
RegisterNumber:2305001028  
*/
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from  sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'],x['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
K=3
Kmeans=KMeans(n_clusters=K)
Kmeans.fit(x)
centroids=Kmeans.cluster_centers_
labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i], label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points, [centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title('K-means Clustring')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![WhatsApp Image 2024-10-26 at 11 00 07_04bdc4bb](https://github.com/user-attachments/assets/0ccdd171-edd1-4332-baca-005b56c39936)
![WhatsApp Image 2024-10-26 at 11 00 31_be4ce23c](https://github.com/user-attachments/assets/a02a2d9b-11c6-4cac-b2d0-99a3f895e89e)
![WhatsApp Image 2024-10-26 at 11 00 44_7a60cb65](https://github.com/user-attachments/assets/f56d30f5-5764-4d67-8962-b8b2b402d95e)
![WhatsApp Image 2024-10-26 at 11 01 06_40957401](https://github.com/user-attachments/assets/96f64834-f215-479c-ac51-3a354721144d)
## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
