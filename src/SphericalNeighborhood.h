#ifndef LAB0_SPHERICALNEIGHBORHOOD_H
#define LAB0_SPHERICALNEIGHBORHOOD_H

#include "Math/float3.h"
#include <Eigen/Eigenvalues>
#include "DataStructures.h"
#include <cstdlib>
#include <chrono>

/// typedef (constant value between iterations in features extractions during subsampling) (c)

class SphericalNeighborhood {
private:
    pointCloudBoost m_cloud, m_neighbors;
    EigenValues m_eigenvalues;
    EigenVectors m_eigenvectors;

    unsigned int m_pointsNum;

    Grid m_grid; /// the grid will be cube, so one variable is enough

    float m_r; /// (c) this variables controls the maximum number of points in our neighborhood. If p is too low, we'll
    /// not have enough points and the features will not be discriminant, but the higher its value is, the longer
    /// computations are.

    /// define scale parameters
    float m_smallestRadius; /// (c) define the radius of the smallest neighborhood
    unsigned int m_scalesNum; /// (c) the number of scales
    float m_ratio; /// (c) ratio between the radius of consecutive neighborhoods

    const vec m_zAxis{0.0, 0.0, 1.0};
    vec m_center;

public:
    void init(const float& r, const unsigned int& scalesNum, const float& smallestRadius, const float& ratio, const pointCloudBoost& cloud);

    void nearestKSearch(const pointCloudBoost &data, const pcl::PointXYZ &searchPoint, int &k,
                        pointCloudBoost &neighbors);

    void nearestKSearchIndices(const pointCloudBoost& data, const pcl::PointXYZ& searchPoint, int& k,
                                                      std::vector<int>& indices);

    void radiusSearch(const pointCloudBoost &data, const pcl::PointXYZ &searchPoint, const float &radius,
                      pointCloudBoost &neighbors);

    void radiusSearchIndices(const pointCloudBoost &data, const pcl::PointXYZ &searchPoint, const float &radius,
                             std::vector<int>& indices, vec& representative);

    float getSmallestRadius();
    float getR();
    Grid getGrid();
    float getRatio();
    double getEigenvaluesSum();
    double getOmnivariance();
    double getEigenentropy();
    double getLinearity();
    double getPlanarity();
    double getSphericity();
    double getCurvatureChange();
    void getVerticality(Verticality& res);
    void getAbsoluteMoment(AbsoluteMoment& res);
    void getVerticalMoment(VerticalMoment& res);
    unsigned int getPointsNumber();
    double getAngle(const vec& p1, const vec& p2);
    double getDistance(const vec& p1, const vec& p2);
    double getAngle(const vec& p1, const vec& p2, const vec& p3);
    vec getCenter();
    void getNeighbors(pointCloudBoost& neighbors);
    void getAllFeatures(GeometricFeatures& feature);

    void setScalesNum(const unsigned int& value);
    void setPointsNum(const unsigned int& value);
    void setSmallestRadius(const float& value);
    void setRatio(const float& value);
    void setR(const float& value);
    void setGridSize(const Grid& grid);
    void setNeighbors(const pointCloudBoost& cloud);
    void setCloud(const pointCloudBoost& cloud);
    void setCenter(const vec& p);

    double dotProduct(const vec& v1, const vec& v2);
    vec crossProduct(const vec& v1, const vec& v2);
    void downsample(pcl::PCLPointCloud2::Ptr& cloud);
    inline void eigendecomposition();
//    void run(std::vector<GeometricFeatures>& geometricFeatures, std::stringstream& fileData, int& categoryIndex);
    void run(std::stringstream& fileData, int& categoryIndex, VecArray& debug);
};

#endif //LAB0_SPHERICALNEIGHBORHOOD_H