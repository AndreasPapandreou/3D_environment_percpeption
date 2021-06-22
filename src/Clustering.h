#ifndef AV_CLUSTERING_H
#define AV_CLUSTERING_H

#include "DataStructures.h"
#include "Helpers.h"
#include "Classifier.h"

#include "Math/float3.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/console/time.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <opencv2/opencv.hpp>

typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;
typedef pcl::PointXYZ Point;
bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance);

class Clustering {
    public:
        int m_iter;
        VecArray lastCurbPoints, curbUp, curbBelow;
        Clustering() {
            m_iter = 0;
        }
        int m_clusters;

        void curbs_clustering(VecArray& points, VecArray& debug);
        void objects_clustering(VecArray &points, VecArray &objectsAboveRoad, calibMat& calib, vec& egoLocation, cv::Mat& img, pcl::PointCloud<PointTypeIO>::Ptr cloud_out, pcl::IndicesClustersPtr clusters);
};
#endif //AV_CLUSTERING_H