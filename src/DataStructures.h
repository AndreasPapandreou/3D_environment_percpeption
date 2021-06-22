#ifndef LAB0_DATASTRUCTURES_H
#define LAB0_DATASTRUCTURES_H

#include <glm/fwd.hpp>
#include <glm/detail/type_vec3.hpp>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "Math/float3.h"
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <eigen3/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <glm/detail/type_vec2.hpp>
#include <Eigen/Geometry>

typedef pcl::PointCloud<pcl::PointXYZ> pointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudBoost; /// Ptr is a typedef to a smart pointer of the Boost library
typedef std::vector<std::pair<float, int>> pairFI; /// Ptr is a typedef to a smart pointer of the Boost library
typedef std::vector<std::tuple<float, int, int>> pairFII; /// Ptr is a typedef to a smart pointer of the Boost library
typedef std::vector<std::tuple<int, float, float>> pairIFF; /// Ptr is a typedef to a smart pointer of the Boost library
typedef std::vector<std::tuple<float, float, int>> pairFFI; /// Ptr is a typedef to a smart pointer of the Boost library
typedef std::vector<std::tuple<int, int, float, vec>> pairIIFV;
typedef std::tuple<int, int, float, vec> varIIFV;
typedef std::vector<std::pair<int, float>> pairIF; /// Ptr is a typedef to a smart pointer of the Boost library
typedef std::vector<std::pair<int, int>> pairII; /// Ptr is a typedef to a smart pointer of the Boost library
typedef std::vector<std::pair<float, float>> pairFF; /// Ptr is a typedef to a smart pointer of the Boost library
typedef std::vector<std::pair<vec, int>> pairVI; /// Ptr is a typedef to a smart pointer of the Boost library

struct Box3D {
    VecArray corners;
    std::string category;
    vec col;

    /// First four corners are the ones facing forward.
    /// The last four are the ones facing backwards.
    ///////////////////////////////////
    ///       5---------------4     ///
    ///      - |             -|     ///
    ///     -  |           -  |     ///
    ///    1-------------0    |     ///
    ///    |   |         |    |     ///
    ///    | 6-----------|----7     ///
    ///    |             |   -      ///
    ///    |             |  -       ///
    ///    -----------------        ///
    ///    2             3          ///
    ///////////////////////////////////

    Box3D(VecArray corners, std::string category, vec col = vec(1.0f, 1.0f, 1.0f)) :
            corners(std::move(corners)), category(std::move(category)), col(col) {};

    Box3D &operator=(Box3D box) {
        corners = box.corners;
        category = box.category;
        col = box.col;
        return *this;
    }
};

typedef std::vector<Box3D> BoxArray;

struct Box3DPoints {
    int topRightF, topLeftF, bottomLeftF, bottomRightF; /// topRightF => TOP RIGHT FRONT
    int topRightB, topLeftB, bottomLeftB, bottomRightB; /// topRightB => TOP RIGHT BACK
};

struct my_Polygon {
    VecArray nodes;
};

struct Annotation
{
    const std::string id;
    const Box3D box;
    const std::string LidarTopPath;
    Annotation(std::string  id, Box3D box, std::string LidarTopPath) :
            id(std::move(id)), box(std::move(box)), LidarTopPath(LidarTopPath) {};
};

struct Sample
{
    const std::string id;
    const std::string LidarTopPath, LidarFrontRightPath, LidarFrontLeftPath;
    /// this vector stores some indices which indicates to values of the vector of Annotations
    const std::vector<unsigned int> AnnIndices;
    const std::vector<std::string> PathToCameraImages;
    const std::vector<unsigned int> RoadSegIndices;
    float yaw, pitch, roll;
    VecArray lidar_top_rotation, ego_pose_rotation;

    /// for lyft
    Sample(std::string  id, std::string LidarTopPath, std::string LidarFrontRightPath, std::string LidarFrontLeftPath,
           std::vector<unsigned int> AnnIndices, std::vector<std::string> PathToCameraImages,
           float yaw, float pitch, float roll, VecArray lidar_top_rotation, VecArray ego_pose_rotation) :
            id(std::move(id)), LidarTopPath(std::move(LidarTopPath)), LidarFrontRightPath(std::move(LidarFrontRightPath)),
            LidarFrontLeftPath(std::move(LidarFrontLeftPath)), AnnIndices(std::move(AnnIndices)),
            PathToCameraImages(std::move(PathToCameraImages)), yaw(yaw), pitch(pitch), roll(roll),
            lidar_top_rotation(std::move(lidar_top_rotation)), ego_pose_rotation(std::move(ego_pose_rotation)) {};

    /// for nuscenes
    Sample(std::string  id, std::string LidarTopPath,
           std::vector<unsigned int> AnnIndices, std::vector<unsigned int> RoadSegIndices) :
            id(std::move(id)), LidarTopPath(std::move(LidarTopPath)), AnnIndices(std::move(AnnIndices)),
            RoadSegIndices(std::move(RoadSegIndices)) {};
};

struct Scene
{
    const std::string id;
    /// this vector stores some indices which indicates to values of the vector of Samples
    const std::vector<unsigned int> SampleIndices;

    Scene(std::string  id, std::vector<unsigned int>  SampleIndices) :
            id(std::move(id)), SampleIndices(std::move(SampleIndices)) {};
};

struct EigenValues
{
    double l1, l2, l3;

    /// TODO add absolute values to eigenvalues => check this
    EigenValues& operator=(const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>::RealVectorType& v) {
        l1 = abs(v(0)); l2 = abs(v(1)); l3 = abs(v(2));
        return *this;
    }
};

struct EigenVectors
{
    vec e1, e2, e3;

    EigenVectors& operator=(const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>::EigenvectorsType& v) {
        e1.x = v.col(0)(0,0); e1.y = v.col(0)(1,0); e1.z = v.col(0)(2,0);
        e2.x = v.col(1)(0,0); e2.y = v.col(1)(1,0); e2.z = v.col(1)(2,0);
        e3.x = v.col(2)(0,0); e3.y = v.col(2)(1,0); e3.z = v.col(2)(2,0);
        return *this;
    }
};

struct Verticality
{
    double firstEigenvectorAxisZ{0.0}, thirdEigenvectorAxisZ{0.0};
    /// the first eigenvector encodes the verticality of linear objects
    /// the third eigenvector encodes the verticality of the normal vector of planar objects

    void clear() {firstEigenvectorAxisZ = 0.0; thirdEigenvectorAxisZ = 0.0;}
};

struct VerticalMoment
{
    double firstOrder{0.0}, secondOrder{0.0};

    void clear() {firstOrder = 0.0; secondOrder = 0.0;}
};

struct AbsoluteMoment
{
    /// firstOrderE1 == first order using the first eigenvector
    double firstOrderE1{0.0}, secondOrderE1{0.0};
    double firstOrderE2{0.0}, secondOrderE2{0.0};
    double firstOrderE3{0.0}, secondOrderE3{0.0};

    void clear() {firstOrderE1 = 0.0; secondOrderE1 = 0.0;
        firstOrderE2 = 0.0; secondOrderE2 = 0.0;
        firstOrderE3 = 0.0; secondOrderE3 = 0.0;}
};

struct GeometricFeatures
{
    int total{11}; /// need to update this if any features is going to be added/subtracted
    double eigenvaluesSum, omnivariance, eigenentropy, linearity, planarity, sphericity, curvatureChange;
    int pointsNumber;
    Verticality verticality;
    VerticalMoment verticalMoment;
    AbsoluteMoment absoluteMoment;

    void clear() {
        eigenvaluesSum = 0.0; omnivariance = 0.0; eigenentropy = 0.0; linearity = 0.0; planarity = 0.0;
        sphericity = 0.0; curvatureChange = 0.0;
        pointsNumber = 0;
        verticality.clear();
        verticalMoment.clear();
        absoluteMoment.clear();
    }

    inline void addToStream(std::stringstream& data, int &categoryIndex) {
        data << categoryIndex << " ";
        data << eigenvaluesSum << " ";
        data << omnivariance << " ";
        data << eigenentropy << " ";
        data << linearity << " ";
        data << planarity << " ";
        data << sphericity << " ";
        data << curvatureChange << " ";
        data << verticality.firstEigenvectorAxisZ << " ";
        data << verticality.thirdEigenvectorAxisZ << " ";
        data << absoluteMoment.firstOrderE1 << " ";
        data << absoluteMoment.firstOrderE2 << " ";
        data << absoluteMoment.firstOrderE3 << " ";
        data << absoluteMoment.secondOrderE1 << " ";
        data << absoluteMoment.secondOrderE2 << " ";
        data << absoluteMoment.secondOrderE3 << " ";
        data << verticalMoment.firstOrder << " ";
        data << verticalMoment.secondOrder << " ";
        data << pointsNumber;
        data <<"\n";
    }

    void print() {
        std::cout << "Sum of eigenvalues : " << eigenvaluesSum << std::endl;
        std::cout << "Omnivariance : " << omnivariance << std::endl;
        std::cout << "Eigenentropy : " << eigenentropy << std::endl;
        std::cout << "Linearity : " << linearity << std::endl;
        std::cout << "Planarity : " << planarity << std::endl;
        std::cout << "Sphericity : " << sphericity << std::endl;
        std::cout << "Change of curvature : " << curvatureChange << std::endl;

        std::cout << "Verticality : " << std::endl;
        std::cout << "firstEigenvectorAxisZ : " << verticality.firstEigenvectorAxisZ << std::endl;
        std::cout << "thirdEigenvectorAxisZ : " << verticality.thirdEigenvectorAxisZ << std::endl;

        std::cout << "Absolute moment : " << std::endl;
        std::cout << "firstOrderAxisX : " << absoluteMoment.firstOrderE1 << std::endl;
        std::cout << "secondOrderAxisX : " << absoluteMoment.secondOrderE1 << std::endl;
        std::cout << "firstOrderAxisY : " << absoluteMoment.firstOrderE2 << std::endl;
        std::cout << "secondOrderAxisY : " << absoluteMoment.secondOrderE2 << std::endl;
        std::cout << "firstOrderAxisZ : " << absoluteMoment.firstOrderE3 << std::endl;
        std::cout << "secondOrderAxisZ : " << absoluteMoment.secondOrderE3 << std::endl;

        std::cout << "Vertical moment : " << std::endl;
        std::cout << "firstOrder : " << verticalMoment.firstOrder << std::endl;
        std::cout << "secondOrder : " << verticalMoment.secondOrder << std::endl;

        std::cout << "Number of points : " << pointsNumber << std::endl;
    }
};

struct Grid
{
    float x{0.0f}, y{0.0f}, z{0.0f};
};

struct SphericalCoordinates
{
    float r{0.0f}, th{0.0f}, f{0.0f};
};

/// A line in 3D space is defined by an origin point and a direction, and extends to infinity in two directions.
/// define parametric equation of line
/// Explanation
/// For two given points P(1,-1,4) and Q(-3,2,-3), we define the direction of vector V from point P to Q,
/// where v = <-4, 3, -7> so the parametric equation is r(t) =  [1 -1 4]' + t[-4 3 -7]'
/// The equations for each axis is :
/// x = 1 - 4t, y = -1 + 3t, z = 4 - 7t
/// Here pos = [1 -1 4] and dir = [-4 3 -7]
struct LineEq
{
    /// Specifies the origin of this line.
    vec pos;
    /// The normalized direction vector of this ray. [similarOverload: pos]
    vec dir;
};

struct calibMat
{
    float fovX, fovY, camCenterX, camCenterY;
//    Eigen::MatrixXd rotation, translation, velodyne_to_cam;
};


#endif //LAB0_DATASTRUCTURES_H