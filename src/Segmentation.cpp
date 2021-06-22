#include "Segmentation.h"

void Segmentation::init(const unsigned int& iter, const unsigned int& numLrp, const float& seedThres, const float& distThres, const unsigned int& num_seg, VecArray& cloud) {
    m_iterations = iter;
    m_numLrp = numLrp;
    m_seedThres = seedThres;
    m_distThres = distThres;
    m_num_seg = num_seg;
    m_cloud = cloud;
}

void Segmentation::setCloud(VecArray& cloud) {
    m_cloud = cloud;
}

void Segmentation::estimateRoad(VecArray& ground, VecArray& noGround, Plane& model, int N) {
    extractInitialSeeds(ground);
    float dist; vec candidate;

    for (unsigned int i=0; i<m_iterations; i++) {
        estimatePlane(ground, model);
        ground.clear(); noGround.clear();

        for (unsigned j=0; j<m_seg_cloud.size(); j++) {
            candidate = vec(m_seg_cloud[j].x, m_seg_cloud[j].y, m_seg_cloud[j].z);
            dist = distFromPlane(candidate, model);
            if (dist < m_distThres) {
                ground.emplace_back(candidate);
            }
            else
                noGround.emplace_back(candidate);
        }
    }
}

void Segmentation::extractInitialSeeds(VecArray& seeds) {
    sortOnHeight();

    /// extract only the first pair from m_sortedCloudWithIndices which are the values of heights
    std::vector<float> sortedCloudZ;
    std::transform(begin(m_sortedCloudZWithIndices), end(m_sortedCloudZWithIndices),
                   std::back_inserter(sortedCloudZ),
                   [](auto const& pair){ return pair.first; });

    if (sortedCloudZ.size() < m_numLrp)
        m_numLrp = sortedCloudZ.size();

    /// compute the lowest point representative taking the average
    m_lrp = accumulate( sortedCloudZ.begin(), sortedCloudZ.begin() + m_numLrp, 0.0)/m_numLrp;

    /// extract initial seeds which are candidate points for the ground
    int index{0};
    for (auto & point : m_sortedCloudZWithIndices) {
        if (point.first < m_lrp + m_seedThres) {
            index = point.second;
            seeds.emplace_back(vec(m_seg_cloud[index].x, m_seg_cloud[index].y, m_seg_cloud[index].z));
        }
    }
}

void Segmentation::sortOnHeight() {
    m_sortedCloudZWithIndices.clear();

    /// Inserting element in pair vector to keep track of previous indexes
    for (unsigned int i=0; i<m_length; i++) {
//        m_sortedCloudZWithIndices.emplace_back(std::make_pair(m_seg_cloud[i].z, i)); /// test
//        m_sortedCloudZWithIndices.emplace_back(std::make_pair( pow(m_cloud->points[i].z + m_cloud->points[i].y, 2), i));
//        m_sortedCloudZWithIndices.emplace_back(std::make_pair( pow(m_cloud->points[i].z + abs(m_cloud->points[i].y), 2), i));
        m_sortedCloudZWithIndices.emplace_back(std::make_pair( m_seg_cloud[i].z + abs(m_seg_cloud[i].y), i)); /// old
    }
    /// Sorting pair vector in increasing order
    sort(m_sortedCloudZWithIndices.begin(), m_sortedCloudZWithIndices.end());
}

void Segmentation::estimatePlane(VecArray& points, Plane& plane) {
    int length = points.size();
    vec center = getCentroid(points);

    vec centered_point;
    Eigen::MatrixXf X(3,length);
    for (int i=0; i<length; i++) {
        centered_point = points[i] - center;
        X(0,i) = centered_point.x;
        X(1,i) = centered_point.y;
        X(2,i) = centered_point.z;
    }

    /// compute the covariance matrix
    Eigen::Matrix3f Cov = X*X.transpose();

    /// compute the singular value decomposition
    Eigen::JacobiSVD<Eigen::MatrixXf> svd;
    svd.compute(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV );
    if (!svd.computeU() || !svd.computeV()) {
        std::cerr << "decomposition error" << std::endl;
    }

    /// extract left singular vectors
    Eigen::Matrix3f U = svd.matrixU();

    plane.normal.x = U(0,2); plane.normal.y = U(1,2); plane.normal.z = U(2,2);
    plane.d = -dotProduct(plane.normal, center);
}

/// \description
/// This function copmutes the centroid of a number of 3d-points
/// \param points
/// \return vec, the centroid
vec Segmentation::getCentroid(const VecArray &points) {
    vec center(0.0f,0.0f,0.0f);
    for (const auto &i : points) {
        center += i;
    }
    return center/(float)points.size();
}

float Segmentation::distFromPlane(vec& point, Plane& model) {
    float numerator = fabs(dotProduct(model.normal, point) + model.d);
    float denominator = sqrt(pow(model.normal.x, 2) + pow(model.normal.y, 2) + pow(model.normal.z, 2));
    return numerator/denominator;
}

/// \description
/// This function computes the dot product of two vec points
/// \param v1
/// \param v2
/// \return double, the result
float Segmentation::dotProduct(const vec& v1, const vec& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

void Segmentation::segmentCloud(VecArray& totalGround, VecArray& totalNoGround, VecArray& allSegmentsV1, std::vector<int>& lengthEachSegV1) {
    pairFI sortedCloudX; pairII initialIndexAndLength;
    VecArray segmentedCloud;
    VecArray ground, noGround;
    Plane modelPlaneV1;

    /// Inserting element in pair vector to keep track of previous indices
    for (unsigned int i=0; i<m_cloud.size(); i++)
        sortedCloudX.emplace_back(std::make_pair(m_cloud[i].x, i));

    /// Sorting pair vector in increasing order
    sort(sortedCloudX.begin(), sortedCloudX.end());
    float minX = sortedCloudX[0].first;
    float maxX = sortedCloudX[sortedCloudX.size()-1].first;
    float interval = abs(minX) + abs(maxX);
    float step = interval/m_num_seg;
    float leftLimit{minX}, rightLimit = minX + step;

    for (unsigned int i=0; i<m_num_seg; i++) {
        m_seg_cloud.clear();

        /// fill with data each segment
        for (unsigned int j=0; j<m_cloud.size(); j++) {
            if ( (m_cloud[j].x >= leftLimit) && (m_cloud[j].x <= rightLimit) ) {
                m_seg_cloud.emplace_back(
                        m_cloud[j].x,
                        m_cloud[j].y,
                        m_cloud[j].z);

                allSegmentsV1.emplace_back(
                        m_cloud[j].x,
                        m_cloud[j].y,
                        m_cloud[j].z);
            }
        }

        leftLimit = rightLimit;
        rightLimit += step;
        lengthEachSegV1.emplace_back(m_seg_cloud.size());
        m_length = m_seg_cloud.size();
        ground.clear(); noGround.clear();
        estimateRoad(ground, noGround, modelPlaneV1, i);
        totalGround.insert(totalGround.end(), ground.begin(), ground.end());
        totalNoGround.insert(totalNoGround.end(), noGround.begin(), noGround.end());
    }
}