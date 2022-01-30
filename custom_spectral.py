import timeit
import pandas as pd
import numpy as np
import os,sys,os.path,shutil
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import platform
import warnings

warnings.filterwarnings('ignore')

def adj_to_laplacian(adj_matrix,return_d_matrix=False):
    adj_copy = np.copy(adj_matrix)
    d_matrix = np.diagflat(np.sum(adj_copy,axis=1))
    laplacian = d_matrix - adj_matrix
    if return_d_matrix:
        return laplacian, d_matrix
    else:
        return laplacian

def check_symmetric(adj_matrix):
    return np.all(np.abs(adj_matrix-adj_matrix.T) < 1e-8)


def random_inital_medoids(eigen_array,k):
    unique_vals,unique_index = np.unique(eigen_array,axis=0,return_index=True)
    medoid_indexes = np.random.choice(unique_index,size=k,replace=False)
    mean_array = eigen_array[medoid_indexes,:]
    return mean_array

def ecludian_distance(eigen_array,means_array):
    eigen_array2 = np.copy(eigen_array)
    means_array2 = np.copy(means_array)
    expanded_eigen_array = np.repeat(eigen_array2[:,:,np.newaxis],means_array2.shape[0],axis=2)
    expanded_means = means_array2[:,:,np.newaxis].T
    mean_labels = np.argmin(np.sum(np.square(expanded_eigen_array - expanded_means),axis=1),axis=1)
    return mean_labels

def re_solve_k_means_eclu(eigen_array,means_array,labels):
    eigen_array2 = np.copy(eigen_array)
    new_means = np.zeros((means_array.shape[0],means_array.shape[1]),dtype=float)
    for i in range(0,means_array.shape[0]):
        inter_cluster_index = np.argwhere(labels == i)[:,0]
        new_means[i,:] = np.average(eigen_array2[inter_cluster_index,:],axis=0)
    return new_means 



def K_means_Spectral(eig_vectors,True_labels_array,k,iter_max=500):
    eig_vectors2 = np.copy(eig_vectors)
    k_eigen_vectors = eig_vectors2[:,0:k].real
    iteration_means = random_inital_medoids(k_eigen_vectors,k=k)
    iteration_labels = ecludian_distance(k_eigen_vectors,iteration_means)
   

    for i in range(0,iter_max):
        new_centroids = re_solve_k_means_eclu(k_eigen_vectors,iteration_means,iteration_labels)
        if np.array_equal(new_centroids,iteration_means):
            break
        else:
            iteration_means = new_centroids
            iteration_labels = ecludian_distance(k_eigen_vectors,iteration_means)
    return np.array(iteration_labels)

def Spectral_Sweep(eig_vectors,True_labels_array,start_k=2,end_k=200,iter_max=500):
    number_of_nodes = eig_vectors.shape[0]
    output_array = []
    eig_vectors2 =np.copy(eig_vectors)
    for j in range(start_k,end_k):
        labels = K_means_Spectral(eig_vectors2,True_labels_array=True_labels_array,k=j,iter_max=iter_max)
        mis_match_list = []
        cluster_sizes = []
        for i in np.unique(labels):
            indicies = np.argwhere(labels==i)[:,0]
            true_labs = True_labels_array[indicies]
            cluster_sizes.append(true_labs.shape[0])
            mode,counts = scipy.stats.mode(true_labs)
            mis_matches = true_labs.shape[0] - counts[0]
            mis_match_rate = mis_matches / true_labs.shape[0]
            mis_match_list.append(mis_match_rate)
        mis_match_array = np.array(mis_match_list)
        cluster_sizes_array = np.array(cluster_sizes)
        avg_j = np.sum(mis_match_array*cluster_sizes_array)/ number_of_nodes
        output_array.append(avg_j)
        #print('------- K Mean Iteration %s Complete -------'% j)
    plt.plot(np.array(output_array))
    plt.xlabel('K Value')
    plt.ylabel('Mis-Match Rate')
    plt.title('Mis-Match Rates for Various Values of K')
    plt.show()
    return output_array

def form_reduced_adjancency_and_labels(edges,nodes):
    nodes_array = np.array(nodes)
    vertex_1_max = edges['vertex_1'].max()
    vertex_1_min = edges['vertex_1'].min()
    vertex_2_max = edges['vertex_2'].max()
    vertex_2_min = edges['vertex_2'].min()
    node_index_high = max(vertex_2_max,vertex_1_max)
    node_index_low = min(vertex_1_min,vertex_2_min)
    adj_matrix = np.zeros((node_index_high,node_index_high))
    for a,b in edges.values:
        adj_matrix[a-1,b-1] = 1
        adj_matrix[b-1,a-1] = 1
    adj_mask = (adj_matrix==0).all(1)
    zero_row_indexes = np.array(np.where(adj_mask==True)[0])
    reduced_nodes_array = np.delete(nodes_array,zero_row_indexes)
    reduced_adj_matrix = adj_matrix[:,~(adj_matrix==0).all(1)]
    reduced_adj_matrix = reduced_adj_matrix[~(reduced_adj_matrix==0).all(1),:]
    assert (adj_matrix.shape[0]) - len(zero_row_indexes) == reduced_adj_matrix.shape[0] == reduced_adj_matrix.shape[1]
    assert check_symmetric(reduced_adj_matrix) == True
    return reduced_adj_matrix, reduced_nodes_array