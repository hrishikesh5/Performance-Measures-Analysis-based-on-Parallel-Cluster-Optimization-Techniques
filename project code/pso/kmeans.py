

import numpy


def calc_sse(centroids: numpy.ndarray, labels: numpy.ndarray, data: numpy.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = numpy.where(labels == i)
        dist = numpy.sum((data[idx] - c)**2)
        distances += dist
    return distances


class KMeans:
   

    def __init__(
            self,
            n_cluster: int,
            init_pp: bool = True,
            max_iter: int = 300,
            tolerance: float = 1e-4,
            seed: int = None):
       

        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_pp = init_pp
        self.seed = seed
        self.centroid = None
        self.SSE = None

    def fit(self, data: numpy.ndarray):
        
        self.centroid = self._init_centroid(data)
        for _ in range(self.max_iter):
            distance = self._calc_distance(data)
            cluster = self._assign_cluster(distance)
            new_centroid = self._update_centroid(data, cluster)
            diff = numpy.abs(self.centroid - new_centroid).mean()
            self.centroid = new_centroid

            if diff <= self.tolerance:
                break

        self.SSE = calc_sse(self.centroid, cluster, data)

    def predict(self, data: numpy.ndarray):
       
        distance = self._calc_distance(data)
        
        cluster = self._assign_cluster(distance)
        
        return cluster

    def _init_centroid(self, data: numpy.ndarray):
       
        if self.init_pp:
            numpy.random.seed(self.seed)
            centroid = [int(numpy.random.uniform()*len(data))]
            for _ in range(1, self.n_cluster):
                dist = []
                dist = [min([numpy.inner(data[c]-x, data[c]-x) for c in centroid])
                        for i, x in enumerate(data)]
                dist = numpy.array(dist)
                dist = dist / dist.sum()
                cumdist = numpy.cumsum(dist)

                prob = numpy.random.rand()
                for i, c in enumerate(cumdist):
                    if prob > c and i not in centroid:
                        centroid.append(i)
                        break
            centroid = numpy.array([data[c] for c in centroid])
        else:
            numpy.random.seed(self.seed)
            idx = numpy.random.choice(range(len(data)), size=(self.n_cluster))
            centroid = data[idx]
        
        return centroid

    def _calc_distance(self, data: numpy.ndarray):
       
        distances = []
        for c in self.centroid:
            distance = numpy.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = numpy.array(distances)
        distances = distances.T
        return distances

    def _assign_cluster(self, distance: numpy.ndarray):
      
        cluster = numpy.argmin(distance, axis=1)
        return cluster

    def _update_centroid(self, data: numpy.ndarray, cluster: numpy.ndarray):
       
        centroids = []
        for i in range(self.n_cluster):
            idx = numpy.where(cluster == i)
            centroid = numpy.mean(data[idx], axis=0)
            centroids.append(centroid)
        centroids = numpy.array(centroids)
        return centroids


if __name__ == "__main__":

    pass
