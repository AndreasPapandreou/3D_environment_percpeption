#ifndef LAB0_SEGMENTATION_H
#define LAB0_SEGMENTATION_H

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <include/MathGeoLib/Geometry/Plane.h>
#include "DataStructures.h"

class Segmentation {
private:
    VecArray m_cloud, m_seg_cloud;
    pairFI m_sortedCloudZWithIndices;
    unsigned int m_length; /// define the length of m_cloud
    unsigned int m_iterations;
    unsigned int m_numLrp; /// number of points to estimate the lowest point representative (lrp)
    float m_lrp; /// define the lowest point representative
    float m_seedThres; /// threshold for points to be considered initial seeds
    float m_distThres; /// threshold distance from the plane
    int m_num_seg; /// define the number of segments
public:
    void init(const unsigned int& iter, const unsigned int& numLrp, const float& seedThres, const float& distThres, const unsigned int& num_seg, VecArray& cloud);
    void extractInitialSeeds(VecArray& seeds);
    void estimatePlane(VecArray& points, Plane& plane);
    void estimateRoad(VecArray& ground, VecArray& noGround, Plane& model, int N);
    void sortOnHeight();
    vec getCentroid(const VecArray &points);
    float distFromPlane(vec& point, Plane& model);
    void segmentCloud(VecArray& totalGround, VecArray& totalNoGround, VecArray& allSegmentsV1, std::vector<int>& lengthEachSegV1);
    void setCloud(VecArray& cloud);
    float dotProduct(const vec& v1, const vec& v2);
    void run(VecArray& totalGround, VecArray& totalNoGround, unsigned int& segments);
    math::vec p0;
};

#endif //LAB0_SEGMENTATION_H