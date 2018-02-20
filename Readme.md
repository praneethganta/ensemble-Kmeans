## K- Means Clustering

 * Implement the k-means clustering algorithm with number of clusters are given to be 3,5 and 7. 
 * Here first we randomly select the centroids from the given data set. Then we use the simple Euclidean distance to calculate the distance between the each data sample and all the centroids selected. 
 * We then select the cluster which has the minimum distance to the data sample and classify that data value to that cluster. 
 * We do this for all the data sample in the data set. Then we recalculate the centroid values. We do this by taking mean of the entire data sample in that particular cluster and make it the new centroid. 
 * Again we repeat the steps until the below conditions is met. During each iteration, we make a note of sum of squared errors (SSE) of distances of samples from the centroids. 
 * We stop the iterations when the change in the SSE of consecutive iterations is less than 0.001 or if number of iterations reach 100. We used ‘pdist2’ for calculating the Euclidean distance. 
 
## Ensemble Classification model

There are 4 parts to this problem. The data set contains images of handwritten digits. We need to classify the data samples and also report the accuracy of each models.
 * First we use the k-nearest neighbor  with k=7
 * We developed a SVM model with a Polynomial Kernel of degree2. We used the same code that we developed for previous assignment
 * We developed a Feedforward neural network with a single hidden layer with 25 neurons
 * We used all the above algorithms to predict the data sample by ensemble method in which we classify a data sample by taking majority vote of the classes predicted using all the above algorithms
