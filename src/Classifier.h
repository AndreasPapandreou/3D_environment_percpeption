#ifndef LAB0_CLASSIFIER_H
#define LAB0_CLASSIFIER_H

#include "iostream"
#include <fstream>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "Helpers.h"
#include "SphericalNeighborhood.h"

typedef pcl::PointXYZI PointTypeIO;

class Classifier {
private:
    std::string m_classifierPath;
public:
    Classifier(std::string classifierPath);
    int run(VecArray& points);
    void run(pcl::PointCloud<PointTypeIO>::Ptr& inner_cloud, pcl::IndicesClustersPtr& clusters, std::vector<int>& categories);
    void run(VecArray& points, std::vector<int>& categories);
    int predict(std::stringstream& trainingData);
    template <typename T>
    T getMin(const std::vector<T>& values);
    template <typename T>
    T getMax(const std::vector<T>& values);
};

#endif //LAB0_CLASSIFIER_H