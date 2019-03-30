from sklearn.cluster import KMeans
import numpy as np

def main():
	input_filename = './yelp_phrase_selected.emb'
	output_filename = './yelp_25_label.txt'
	with open(input_filename) as input:
		vector = []
		words = []
		for line in input.readlines():
			value = line.split(' ')
			words.append(value[0])
			value = value[1:101]
			value = np.array(value).astype(np.float64)
			vector.append(value)
			
		arr = np.array(vector)
		kmeans = KMeans(n_clusters=25, random_state=0).fit(arr)
	
	with open(output_filename, 'w') as output:
		clusters = []
		for i in range(arr.shape[0]):
			cluster = []
			cluster.append(words[i])
			cluster.append(kmeans.labels_[i])
			clusters.append(cluster)

		results = sorted(clusters, key=lambda k: k[1])
		for i in range(len(results)):
			output.write(results[i][0] + ' ' + str(results[i][1]) + '\n')


if __name__ == "__main__":
    main()

