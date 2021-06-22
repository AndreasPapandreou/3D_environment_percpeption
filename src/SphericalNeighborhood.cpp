#include "SphericalNeighborhood.h"

using namespace std::chrono;

void SphericalNeighborhood::init(const float& r, const unsigned int& scalesNum, const float& smallestRadius, const float& ratio,
                                 const pointCloudBoost& cloud) {
    m_r = r;
    m_scalesNum = scalesNum;
    m_smallestRadius = smallestRadius;
    m_ratio = ratio;
    m_cloud = cloud;
    m_pointsNum = cloud->width*cloud->height;
}

/// K nearest neighbor search
/// TODO optimize code handling cases where there is no neighbor
void SphericalNeighborhood::nearestKSearch(const pointCloudBoost& data, const pcl::PointXYZ& searchPoint, int& k,
                                            pointCloudBoost& neighbors) {

//    std::cout << "nearestKSearch" << std::endl;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(data);

    std::vector<int> pointIdxNKNSearch(k);
    std::vector<float> pointNKNSquaredDistance(k);

    /// increase k by one because  the first neighbor in pcl is equall to the serach point
    kdtree.nearestKSearch (searchPoint, k+1, pointIdxNKNSearch, pointNKNSquaredDistance);
    int NumNeighbors = pointIdxNKNSearch.size();

//    /// reserve space for neighbors
//    neighbors->width = NumNeighbors;
//    neighbors->height = 1;
//    neighbors->points.resize (neighbors->width * neighbors->height);
//
//    if ( NumNeighbors > 0 ) {
//        for (unsigned int i = 0; i < pointIdxNKNSearch.size(); ++i) {
//            neighbors->points[i].x = data->points[pointIdxNKNSearch[i]].x;
//            neighbors->points[i].y = data->points[pointIdxNKNSearch[i]].y;
//            neighbors->points[i].z = data->points[pointIdxNKNSearch[i]].z;
////            std::cout <<" squared distance: " << pointNKNSquaredDistance[i] << std::endl;
//        }
//    }

    if(NumNeighbors <= 1) { /// it means that there are no neighbors between this radius
        neighbors->width = 0; neighbors->height = 0;
    }
    else {
        /// reserve space for neighbors
        neighbors->width = NumNeighbors;
        neighbors->height = 1;
        neighbors->width -= 1; /// remove the search point
        neighbors->points.resize (neighbors->width * neighbors->height);

        /// start from 1, because the neighbors->points[0] is equal with the search point
        for (unsigned int i = 1; i < NumNeighbors; i++) {
            neighbors->points[i-1].x = data->points[pointIdxNKNSearch[i]].x;
            neighbors->points[i-1].y = data->points[pointIdxNKNSearch[i]].y;
            neighbors->points[i-1].z = data->points[pointIdxNKNSearch[i]].z;
        }
    }
}

/// K nearest neighbor search
/// TODO optimize code handling cases where there is no neighbor
void SphericalNeighborhood::nearestKSearchIndices(const pointCloudBoost& data, const pcl::PointXYZ& searchPoint, int& k,
                                            std::vector<int>& indices) {

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(data);

    std::vector<int> pointIdxNKNSearch(k);
    std::vector<float> pointNKNSquaredDistance(k);

    /// increase k by one because  the first neighbor in pcl is equall to the serach point
    kdtree.nearestKSearch (searchPoint, k+1, pointIdxNKNSearch, pointNKNSquaredDistance);
    int NumNeighbors = pointIdxNKNSearch.size();

    if(NumNeighbors <= 1) { /// it means that there are no neighbors between this radius
        return;
    }
    else {
        /// start from 1, because the neighbors->points[0] is equal with the search point
        for (unsigned int i=1; i < NumNeighbors; i++) {
            indices.emplace_back(pointIdxNKNSearch[i]);
        }
    }
}

/// Neighbors within radius search => it returns the number of neighbors of a given search points. The first point of
/// neighbors is the same search point
/// TODO optimize code handling cases where there is no neighbor
void SphericalNeighborhood::radiusSearch(const pointCloudBoost& data, const pcl::PointXYZ& searchPoint, const float& radius,
                                        pointCloudBoost& neighbors) {

//    std::cout << "radiusSearch" << std::endl;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(data);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
    int NumNeighbors = pointIdxRadiusSearch.size();

    if(NumNeighbors <= 1) { /// it means that there are no neighbors between this radius
        neighbors->width = 0; neighbors->height = 0;
    }
    else {
        /// reserve space for neighbors
        neighbors->width = NumNeighbors;
        neighbors->height = 1;
        neighbors->width -= 1; /// remove the search point
        neighbors->points.resize (neighbors->width * neighbors->height);

        /// start from 1, because the neighbors->points[0] is equal with the search point
        for (unsigned int i = 1; i < NumNeighbors; i++) {
            neighbors->points[i-1].x = data->points[pointIdxRadiusSearch[i]].x;
            neighbors->points[i-1].y = data->points[pointIdxRadiusSearch[i]].y;
            neighbors->points[i-1].z = data->points[pointIdxRadiusSearch[i]].z;
        }
    }
}

/// Neighbors within radius search => it returns the number of neighbors of a given search points. The first point of
/// neighbors is the same search point
/// TODO optimize code handling cases where there is no neighbor
void SphericalNeighborhood::radiusSearchIndices(const pointCloudBoost& data, const pcl::PointXYZ& searchPoint, const float& radius,
                                        std::vector<int>& indices, vec& representative) {

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(data);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
    int NumNeighbors = pointIdxRadiusSearch.size();

//    if(NumNeighbors <= 1)
//        return;
//    else {
        /// start from 1, because the neighbors->points[0] is equal with the search point
        for (unsigned int i = 0; i < NumNeighbors; i++) {
            indices.emplace_back(pointIdxRadiusSearch[i]);
            representative.x += data->points[pointIdxRadiusSearch[i]].x;
            representative.y += data->points[pointIdxRadiusSearch[i]].y;
            representative.z += data->points[pointIdxRadiusSearch[i]].z;
        }
//    }
    representative /= (NumNeighbors);


    /// it means that there are no neighbors between this radius
//    if(NumNeighbors <= 1)
//        return;
//    else {
//        /// start from 1, because the neighbors->points[0] is equal with the search point
//        for (unsigned int i = 1; i < NumNeighbors; i++) {
//            indices.emplace_back(pointIdxRadiusSearch[i]);
//            representative.x += data->points[pointIdxRadiusSearch[i]].x;
//            representative.y += data->points[pointIdxRadiusSearch[i]].y;
//            representative.z += data->points[pointIdxRadiusSearch[i]].z;
//        }
//    }
//    representative /= (NumNeighbors-1);
}

float SphericalNeighborhood::getSmallestRadius() {
    return m_smallestRadius;
}

float SphericalNeighborhood::getR() {
    return m_r;
}

Grid SphericalNeighborhood::getGrid() {
    return m_grid;
}

float SphericalNeighborhood::getRatio() {
    return m_ratio;
}

/// TODO checn nan/inf
double SphericalNeighborhood::getEigenvaluesSum() {
    double res = m_eigenvalues.l1 + m_eigenvalues.l2 + m_eigenvalues.l3;
    if (isnan(res))
        return 0.0;
    return res;
}

/// TODO checn nan/inf
double SphericalNeighborhood::getOmnivariance() {
    double res = pow((m_eigenvalues.l1 * m_eigenvalues.l2 * m_eigenvalues.l3), 1.0/3.0);
    if (isnan(res))
        return 0.0;
    return res;
}

/// TODO checn nan/inf
double SphericalNeighborhood::getEigenentropy() {
    double res = -(m_eigenvalues.l1 * log(m_eigenvalues.l1) +
                   m_eigenvalues.l2 * log(m_eigenvalues.l2) +
                   m_eigenvalues.l3 * log(m_eigenvalues.l3));
    if (isnan(res))
        return 0.0;
    return res;
}

/// TODO checn nan/inf
double SphericalNeighborhood::getLinearity() {
    double res = (m_eigenvalues.l1 -m_eigenvalues.l2)/m_eigenvalues.l1;
    if (isnan(res) || isinf(res))
        return 0.0;
    return res;
}

/// TODO checn nan/inf
double SphericalNeighborhood::getPlanarity() {
    double res = (m_eigenvalues.l2 -m_eigenvalues.l3)/m_eigenvalues.l1;
    if (isnan(res) || isinf(res))
        return 0.0;
    return res;
}

/// TODO checn nan/inf
double SphericalNeighborhood::getSphericity() {
    double res = (m_eigenvalues.l3/ m_eigenvalues.l1);
    if (isnan(res) || isinf(res))
        return 0.0;
    return res;
}

/// TODO checn nan/inf
double SphericalNeighborhood::getCurvatureChange() {
    double res = (m_eigenvalues.l3/(m_eigenvalues.l1 + m_eigenvalues.l2 + m_eigenvalues.l3));
    if (isnan(res) || isinf(res))
        return 0.0;
    return res;
}

/// TODO checn nan/inf
void SphericalNeighborhood::getVerticality(Verticality& res) {
    res.clear();
    res.firstEigenvectorAxisZ = abs(M_PI/2 - getAngle(m_eigenvectors.e1, m_zAxis));
    res.thirdEigenvectorAxisZ = abs(M_PI/2 - getAngle(m_eigenvectors.e3, m_zAxis));
}

/// TODO checn nan/inf
void SphericalNeighborhood::getAbsoluteMoment(AbsoluteMoment& res) {
    int numNeighbors = m_neighbors->width * m_neighbors->height;
    res.clear();
    vec p;
    for (unsigned int i=0; i<numNeighbors; i++) {
        p.x = m_neighbors->points[i].x; p.y = m_neighbors->points[i].y; p.z = m_neighbors->points[i].z;

        /// first order moment
        res.firstOrderE1 += abs(dotProduct(p-m_center, m_eigenvectors.e1));
        res.firstOrderE2 += abs(dotProduct(p-m_center, m_eigenvectors.e2));
        res.firstOrderE3 += abs(dotProduct(p-m_center, m_eigenvectors.e3));

        /// second order moment
        res.secondOrderE1 += abs(pow(dotProduct(p-m_center, m_eigenvectors.e1), 2));
        res.secondOrderE2 += abs(pow(dotProduct(p-m_center, m_eigenvectors.e2), 2));
        res.secondOrderE3 += abs(pow(dotProduct(p-m_center, m_eigenvectors.e3), 2));
    }
    res.firstOrderE1 /= m_pointsNum; res.firstOrderE2 /= m_pointsNum; res.firstOrderE3 /= m_pointsNum;
    res.secondOrderE1 /= m_pointsNum; res.secondOrderE2 /= m_pointsNum; res.secondOrderE3 /= m_pointsNum;
}

/// TODO checn nan/inf
void SphericalNeighborhood::getVerticalMoment(VerticalMoment& res) {
    int numNeighbors = m_neighbors->width * m_neighbors->height;
    res.clear();
    vec p;
    for (unsigned int i=0; i<numNeighbors; i++) {
        p.x = m_neighbors->points[i].x; p.y = m_neighbors->points[i].y; p.z = m_neighbors->points[i].z;
        /// first order
        res.firstOrder += dotProduct(p - m_center, m_zAxis);
        /// second order
        res.secondOrder += pow(dotProduct(p - m_center, m_zAxis),2);
    }
    res.firstOrder /= m_pointsNum; res.secondOrder /= m_pointsNum;
}

unsigned int SphericalNeighborhood::getPointsNumber() {
    return m_pointsNum;
}

/// compute the angle of two 3d points, it returns the inverse cosine of a number (argument) in degrees.
double SphericalNeighborhood::getAngle(const vec& p1, const vec& p2) {
    double numerator = dotProduct(p1, p2);
    double denominator = p1.Length()*p2.Length();
    double radian = acos(numerator/denominator); /// radian lies between –\pi to +\pi
    return radian * 180.0 / M_PI; /// convert radian to degree
}

vec SphericalNeighborhood::getCenter() {
    return m_center;
}

void SphericalNeighborhood::getNeighbors(pointCloudBoost& neighbors) {
    *neighbors = *m_neighbors;
}

///      (p3)----
///         -
///         -
///        -(p1)
///       -
///   ---(p2)
/// compute the angle of three 3d points, it returns the inverse cosine of a number (argument) in degrees.
double SphericalNeighborhood::getAngle(const vec& p1, const vec& p2, const vec& p3) {
    double numerator = (pow(getDistance(p1, p2), 2) + pow(getDistance(p1, p3), 2) - pow(getDistance(p2, p3), 2));
    double denominator =  2*getDistance(p1, p2)*getDistance(p1, p3);
    double radian = acos(numerator/denominator); /// radian lies between –\pi to +\pi
    return radian * 180.0 / M_PI; /// convert radian to degree
}

double SphericalNeighborhood::getDistance(const vec& p1, const vec& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}

void SphericalNeighborhood::getAllFeatures(GeometricFeatures& feature) {
    feature.eigenvaluesSum = getEigenvaluesSum(); /// or this->getEigenvaluesSum
    feature.omnivariance = getOmnivariance();
    feature.eigenentropy = getEigenentropy();
    feature.linearity = getLinearity();
    feature.planarity = getPlanarity();
    feature.sphericity = getSphericity();
    feature.curvatureChange = getCurvatureChange();
    getVerticality(feature.verticality);
    getAbsoluteMoment(feature.absoluteMoment);
    getVerticalMoment(feature.verticalMoment);
    feature.pointsNumber = getPointsNumber();
}

/// \description
/// This function computes the dot product of two vec points
/// \param v1
/// \param v2
/// \return double, the result
double SphericalNeighborhood::dotProduct(const vec& v1, const vec& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

/// \description
/// This function computes the dot product of two vec points
/// \param v1
/// \param v2
/// \return vec, the result
vec SphericalNeighborhood::crossProduct(const vec& v1, const vec& v2) {
    vec res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.x;
    return res;
}

void SphericalNeighborhood::downsample(pcl::PCLPointCloud2::Ptr& cloud) {
    /// Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    /// downsample
    sor.setInputCloud(cloud);
    sor.setLeafSize (m_grid.x, m_grid.y, m_grid.z);
    sor.filter(*cloud);
}

void SphericalNeighborhood::setScalesNum(const unsigned int& value) {
    m_scalesNum = value;
}

void SphericalNeighborhood::setPointsNum(const unsigned int& value) {
    m_pointsNum = value;
}

void SphericalNeighborhood::setSmallestRadius(const float& value) {
    m_smallestRadius = value;
}

void SphericalNeighborhood::setRatio(const float& value) {
    m_ratio = value;
}

void SphericalNeighborhood::setR(const float& value) {
    m_r = value;
}

void SphericalNeighborhood::setGridSize(const Grid& grid) {
    m_grid = grid;
}

void SphericalNeighborhood::setNeighbors(const pointCloudBoost& cloud) {
    m_neighbors = cloud;
}

void SphericalNeighborhood::setCloud(const pointCloudBoost& cloud) {
    m_cloud = cloud;
}

void SphericalNeighborhood::setCenter(const vec& p) {
    m_center = p;
}

inline void SphericalNeighborhood::eigendecomposition() {
    /// TODO check is covariance matrix is valid

    int numNeighbors = m_neighbors->width * m_neighbors->height;
    Eigen::MatrixXf X(3, numNeighbors);
//    std::cout << "center = (" << m_center.x << ", " << m_center.y << ", " << m_center.z << ")" << std::endl;
//    std::cout << "numNeighbors = " << numNeighbors << std::endl;

    for (unsigned int i=0; i<numNeighbors; i++) {
//        std::cout << "neigh = (" << m_neighbors->points[i].x << ", " << m_neighbors->points[i].y << ", " << m_neighbors->points[i].z << ")" << std::endl;
        X(0,i) = m_neighbors->points[i].x - m_center.x;
        X(1,i) = m_neighbors->points[i].y - m_center.y;
        X(2,i) = m_neighbors->points[i].z - m_center.z;
    }

    /// compute the covariance matrix
    Eigen::Matrix3f Cov = X*X.transpose();
    Cov /= numNeighbors;
//    std::cout << "cov = " << Cov << std::endl;

    /// compute eigenvectors and eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(Cov);
    m_eigenvectors = es.eigenvectors();
    m_eigenvalues = es.eigenvalues();

//    std::cout << "The eigenvalues of A are : (" << m_eigenvalues.l1 << ", "  << m_eigenvalues.l2 << ", "  << m_eigenvalues.l3 << ")" << std::endl;
//    std::cout << "The matrix of eigenvectors, V, is:" << m_eigenvectors << std::endl;
}

void SphericalNeighborhood::run(std::stringstream& fileData, int& categoryIndex, VecArray& debug) {
    pointCloudBoost neighbors(new pointCloud), cloud(new pointCloud);
    GeometricFeatures feature;
    pcl::PointXYZ searchPoint(m_center.x, m_center.y, m_center.z);
    bool lastDownsampling{false};
    float radius;

    *cloud = *m_cloud; /// data copy
    for (unsigned int j=0; j<m_scalesNum; j++) {
        /// define new parameters
        radius = float(m_smallestRadius*pow(m_ratio,j));
        m_grid.x = radius/m_r; m_grid.y = m_grid.x; m_grid.z = m_grid.x; /// set grid as cube

        neighbors->clear();
        radiusSearch(cloud, searchPoint, radius, neighbors);
        m_neighbors = neighbors; /// pointer copy

        VecArray tmpPoints;
        int scale_ind{3};
        if (j == scale_ind) {
            for (auto &n : *m_neighbors) {
                tmpPoints.emplace_back(vec(n.x, n.y, n.z));
                debug.emplace_back(vec(n.x, n.y, n.z));
                debug.emplace_back(vec(1.0f, 0.0f, 0.0f));
            }
        }

        int NumNeighbors = neighbors->width;
        if (NumNeighbors == 0)
            continue;

        /// eigendecomposition
        eigendecomposition();

        /// extract features
        getAllFeatures(feature);

        m_pointsNum = neighbors->width;

        /// store training data
        feature.addToStream(fileData, categoryIndex);

        /// downsample
        lastDownsampling = (j == m_scalesNum-1);
        if (!lastDownsampling) {
            pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2 ()); /// generate PCLPointCloud2
            pcl::toPCLPointCloud2(*cloud, *cloud_filtered); /// convert PCLPointCloud to PCLPointCloud2

            if (j == scale_ind) {
                for (auto & p : cloud->points) {
                    bool flag{true};
                    for (auto &v : tmpPoints){
                        if ((v.x == p.x) && (v.y == p.y) && (v.z == p.z))
                            flag = false;
                    }
                    if (flag) {
                        debug.emplace_back(vec(p.x, p.y, p.z));
                        debug.emplace_back(vec(1.0f, 1.0f, 1.0f));
                    }
                }
            }

            downsample(cloud_filtered); /// downsample
            pcl::fromPCLPointCloud2(*cloud_filtered, *cloud); /// convert PCLPointCloud2 to PCLPointCloud
        }
    }
}