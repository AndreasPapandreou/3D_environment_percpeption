#ifndef LAB0_CURBDETECTION_H
#define LAB0_CURBDETECTION_H

#include <vector>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_line.h>

#include "SphericalNeighborhood.h"
#include "DataStructures.h"
#include "polynomial_equation.h"

class CurbDetection {
private:
    VecArray m_lidarPoints, m_projections; /// m_projections[0] refers to m_lidarPoints[0], etc
    std::vector<SphericalCoordinates> m_sphericalPoints;
    int m_rings; /// define the number of rings
    glm::vec3 m_egoPos;
    std::vector<float> m_coeffsAbove, m_coeffsBelow, m_coeffs;
    int m_bins; /// bining angles of spherical points in order to extract rings
public:
    bool m_occlusion, m_left_occlusion, m_right_occlusion;

    void initialize(const VecArray& points, const int& rings, const glm::vec3& egoPos);
    void projectionToXY(VecArray& projections, vec& projectionPoint);
    pairIFF* extractRings(float& initialRadius, const float& radiusIncrement, std::unordered_map<int, int>& idToRing, bool& occlusion, VecArray& testPoints, unsigned int& occlusionParam);
    void detection(pairIFF*& dataPerRing, VecArray& curbPoints, VecArray& testPoints, pairIIFV& results);
    bool filter_stage_1(std::vector<vec>& points, int ringId);
    float variance(std::vector<float> values);
    Eigen::VectorXd linear_ransac(VecArray& curbPoints, double& threshold, bool& success);
    Eigen::VectorXd polynomial_ransac(VecArray& curbPoints, double& threshold, bool& success);
    SphericalCoordinates convertToSpherical(const vec& point);
    void generateColors(const int& num, VecArray& cols);
    float diff(const vec&p1, const vec&p2);
    double dotProduct(const vec& v1, const vec& v2);
    void pca(VecArray& points, Eigen::Matrix3f& eigen_vectors, Eigen::Vector3f& eigen_values, VecArray& testPoints, vec currentP);
    bool lineFitting(std::vector<float>& x, std::vector<float>& y, const int& order, std::vector<float>& coeffs);
    void extractRoadLines(VecArray& curbPoints, VecArray& roadLine, float meanDistValue);
    bool existInVec(VecArray values, vec value);
    bool sameVec(vec p1, vec p2);
    bool isCurved(VecArray& points, float& t);
    void findCandidateCurb(std::vector<std::pair<int, float>>& indices_angles, std::vector<vec>*& candidates, unsigned int& curbPointsNum, unsigned int& regionId, unsigned int& ringId, int& candId, VecArray& testPoints, bool& found, vec& candidate, varIIFV& res);
    void predict(pairIIFV& data);

    /// multi-feature loose-threshold layered methods to extract candidate points
    Eigen::VectorXd getRoadEquation(bool flag, VecArray& points, bool& success);

    void setProjectionPoints(const VecArray& points);
    void getRoadPoints(bool& isCurvedUp, bool& isCurvedBelow, Eigen::VectorXd roadUp, Eigen::VectorXd roadBelow, VecArray& curbUp, VecArray& curbBelow, VecArray& allPoints, VecArray& roadPointsVector);

    vec getPointWithMinX(const VecArray& values);
    vec getPointWithMaxX(const VecArray& values);

    int getBins();
    float getAngle(const vec& p1, const vec& p2, const vec& p3);

    template <typename T>
    T getMin(const std::vector<T>& values);
    template <typename T>
    T getMax(const std::vector<T>& values);
    template <typename T>
    T getMean(const std::vector<T>& values);
};

#endif //LAB0_CURBDETECTION_H