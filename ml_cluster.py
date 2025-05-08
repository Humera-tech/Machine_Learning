from sklearn.cluster import KMeans
import numpy as np 
import matplotlib.pyplot as plt

x=np.array([[10,550],[11,600],[20,680],[23,695],[25,750],[30,790],[35,830],[40,895],[45,910]])

kmeans=KMeans(n_clusters=3,random_state=0)
kmeans.fit(x)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Labels:", labels)
print("Centers:", centers)



plt.scatter(x[:,0],x[:,1],c=labels,cmap="viridis",s=100)
plt.scatter(centers[:,0],centers[:,1],c='red',marker='X',s=100)

plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Student Marks")
plt.legend()
plt.grid(True)
plt.show()

