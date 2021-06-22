#include "CurbDetection.h"

#define CARLA_DATA 0
#define LYFT_DATA 1

/// Driver function to sort the vector elements by second element of pairs
bool sortbysec(const std::pair<int,float> &a, const std::pair<int,float> &b)
{
    return (a.second < b.second);
}

/// Driver function to sort the vector elements by first element of pair in descending order
bool sortinrev(const std::pair<float,int> &a, const std::pair<float,int> &b)
{
    return (a.first > b.first);
}

void CurbDetection::initialize(const VecArray& points, const int& rings, const glm::vec3& egoPos) {
    m_lidarPoints = points;
    m_rings = rings;
    m_egoPos = egoPos;
}

/// m_projections and m_lidarPoints are related 1 by 1, e.g m_projections[0] is related to m_lidarPoints[0]
/// projectionPoint indicates the points from which the projection starts
void CurbDetection::projectionToXY(VecArray& projections, vec& projectionPoint) {
    LineEq line;
    vec newPoint;
    float t;

    line.pos = projectionPoint;
    for (auto & point : m_lidarPoints) {
        line.dir = point - line.pos;

        /// find the value of dependent variable (t) solving the equation of z=0
        t = -line.pos.z/line.dir.z;

        newPoint.x = line.pos.x + t*line.dir.x;
        newPoint.y = line.pos.y + t*line.dir.y;
        newPoint.z = line.pos.z + t*line.dir.z;

        newPoint -= projectionPoint; /// remove the lidarPos from each point in order for the new center to be equal with the old one
        projections.emplace_back(newPoint);
    }
};

/// This function exracts rings from lidar data. The structure std::vector<int>* ringIndices stores the ring id in
/// which each point belongs.
/// For example : [ 0  => [ 10 20 ]
///                1 ] => [ 40 50 ]
/// That means that the ring with index 0 consist of points from VecArray points (which is given as input) with indices
/// 10 and 20 and the ring with index 1 consists of points indices 40 and 50.
/// The structure of m_sphericalPoints stores the spherical point of each projection point, e.g m_sphericalPoints[0] is related to m_projections[0]
pairIFF* CurbDetection::extractRings(float& initialRadius, const float& radiusIncrement, std::unordered_map<int, int>& idToRing, bool& occlusion, VecArray& testPoints,
                                     unsigned int& occlusionParam) {
    SphericalCoordinates scPoint;
    pairFI shR; /// define the radius of each spherical point with an index

    /// convert points from cartesian to spherical coordinates
    for (unsigned int i=0; i<m_projections.size(); i++) {
        scPoint = convertToSpherical(m_projections[i]);
        shR.emplace_back(std::make_pair(scPoint.r, i));
        m_sphericalPoints.emplace_back(scPoint);
    }

    sort(shR.begin(), shR.end()); /// sort shR in increasing order of first element

    /// binning angles of spherical points

    /// set bins bigger than 360 in order not to be created bins with zero points in the first rings and more and more
    /// points in the next rings. If that happens then these data won't be set to the valid rings.
    m_bins = 365; ///TODO test this parameter
//    m_bins = 90; ///TODO test this parameter

    int step{1}, id;
//    int step{4}, id;

    auto* dataPerAngle = new pairFFI[m_bins]; /// store the indices of points that belong to each angle from smallest to bigger radius,
    /// each item includes [radius, angle, index]

    pairIF binsRadius; /// pair = (binId, smallestRadius) => helps me process data of specific angle that has the smallest radius
    for (unsigned int i=0; i<m_bins; i++)
        binsRadius.emplace_back(i, 10000.0f);

    VecArray ringsCols;
    generateColors(m_bins, ringsCols);
    /// define left and right bin angle to check for occlusion
    int r_range{330}, l_range{70}, max_angle{360};

    for (auto & i : shR) {
        id = i.second;
        for (auto j=m_bins-1; j>=0; j--) {
            if (m_sphericalPoints[id].th >= step * j - step) {

//                if (m_sphericalPoints[id].th < 90.0f || m_sphericalPoints[id].th > 270.0f) {
//                    if (m_sphericalPoints[id].r >= 0.15f) {
//                        dataPerAngle[j].emplace_back(m_sphericalPoints[id].r, m_sphericalPoints[id].th,
//                                                     id); /// store points of each angle from smallest to bigger radius
//                        if (m_sphericalPoints[id].r < binsRadius[j].second)
//                            binsRadius[j].second = m_sphericalPoints[id].r;
//                        break;
//                    } /// to delete
//                }
//                else {
//                    if (m_sphericalPoints[id].r >= 0.187f) {
//                        dataPerAngle[j].emplace_back(m_sphericalPoints[id].r, m_sphericalPoints[id].th, id); /// store points of each angle from smallest to bigger radius
//                        if (m_sphericalPoints[id].r < binsRadius[j].second)
//                            binsRadius[j].second = m_sphericalPoints[id].r;
//                        break;
//                    } /// to delete
//                }



                testPoints.emplace_back(m_lidarPoints[id]);
                testPoints.emplace_back(ringsCols[j]);


                        dataPerAngle[j].emplace_back(m_sphericalPoints[id].r, m_sphericalPoints[id].th,
                                                     id); /// store points of each angle from smallest to bigger radius
                        if (m_sphericalPoints[id].r < binsRadius[j].second)
                            binsRadius[j].second = m_sphericalPoints[id].r;
                        break;


            }
        }
    }

    sort(binsRadius.begin(), binsRadius.end(), sortbysec); /// sort by second element of pair in increasing order

    auto* dataPerRing = new pairIFF[m_rings]; /// store the indices of points, its angle and radii referring to the lidar sensor per ring
    SphericalNeighborhood sp;
    pointCloudBoost cloud(new pointCloud);
    pcl::PointXYZ searchP;
    std::unordered_map<int, int> hash, cloudLidar;  /// cloudLidar => (cloudId, lidarId)
    std::vector<int> indices;
    int ringId{0}, binIndex;
    float threshold;
    float r;
    vec newRepresentative, oldRepresentative;

    int occlusion_counter{0};
    /// get representative distance between rings
    for (int item=0; item<m_bins; item++) {
        hash.clear();
        cloudLidar.clear();
        binIndex = binsRadius[item].first;

        /// resize space for point cloud
        cloud->width = dataPerAngle[binIndex].size(); cloud->height = 1;
        cloud->points.resize (cloud->width * cloud->height);
        /// fill point cloud with data
        for (unsigned int m=0; m<cloud->points.size(); m++) {
            id = std::get<2>(dataPerAngle[binIndex][m]);
            cloud->points[m].x = m_projections[id].x; cloud->points[m].y = m_projections[id].y; cloud->points[m].z = m_projections[id].z;
            cloudLidar[m] = id;
        }

        /// correct number up to here

        ringId = -1;

        #if CARLA_DATA == 1
            r = 0.001f;
            threshold = 0.0015f;
        #endif
        #if LYFT_DATA == 1
            r = 0.01f; /// old
//            r = 0.001f; /// test

            threshold = 0.0015f; /// old
//            threshold = 0.00015f; /// test
        #endif

        /// set oldRepresentative to zero
        oldRepresentative.x = 0.0f; oldRepresentative.y = 0.0f; oldRepresentative.z = 0.0f;

        float prev_diff;
        bool first_time{true}, flag{true};

        for (std::tuple myTuple : dataPerAngle[binIndex]) {
            id = std::get<2>(myTuple);
            if (hash[id] == 0) {
                searchP.x = m_projections[id].x; searchP.y = m_projections[id].y; searchP.z = m_projections[id].z;
                indices.clear();
                newRepresentative.Set(0.0f, 0.0f, 0.0f);

                #if CARLA_DATA == 1
                     r = r + (ringId+2)*0.00001f; /// ringId starts from -1
                #endif

                # if LYFT_DATA == 1
                    r = r + (ringId+2)*0.0001f; /// old
//                    r = r + (ringId+2)*0.001f; /// test
                #endif

                sp.radiusSearchIndices(cloud, searchP, r, indices, newRepresentative);

                /// ************************************************************************
                /// check for occlusion
                /// ************************************************************************
                /// add flag in order to get in once per bin
                if (ringId < m_rings/2) {
                    if (flag) {
                        if (binIndex <= max_angle) {
                            if (binIndex <= l_range || binIndex >= r_range) {
                                if (first_time) {
                                    prev_diff = diff(newRepresentative, oldRepresentative);
                                    first_time = false;
                                }
                                else {
//                                    if ( diff(newRepresentative, oldRepresentative) > 20.0f*prev_diff) {
                                    if ( diff(newRepresentative, oldRepresentative) > occlusionParam*prev_diff) {
                                        occlusion_counter++;

                                        /// if there is occlusion left of the car
                                        if (binIndex <= l_range)
                                            m_left_occlusion = true;
                                        /// if there is occlusion right to the car
                                        if (binIndex >= r_range)
                                            m_right_occlusion = true;

                                        flag = false;
                                    }
                                    prev_diff = diff(newRepresentative, oldRepresentative);
                                }
                            }
                        }
                    }
                }
                /// ************************************************************************

                if (diff(newRepresentative, oldRepresentative) > threshold) {
                    ringId += 1;
                    if (ringId >= m_rings)
                        ringId = m_rings-1;
                }
                oldRepresentative = newRepresentative;

                std::tuple<int, float, float> myTuple2;

                for (int & index : indices) {
                    if (hash[cloudLidar[index]] == 0) {
                        std::get<0>(myTuple2) = cloudLidar[index]; /// get index
                        std::get<1>(myTuple2) = std::get<1>(myTuple); /// get angle
                        std::get<2>(myTuple2) = std::get<0>(myTuple); /// get radius
                        dataPerRing[ringId].emplace_back(myTuple2);
                        idToRing[cloudLidar[index]] = ringId;
                        hash[cloudLidar[index]] = 1;
                    }
                }
            }
        }
    }

//    std::cout << "occlusion_counter = " << occlusion_counter << std::endl;

    /// TODO check for the threshold of occlusions..

    #if CARLA_DATA == 1
        if (occlusion_counter >= 20) {
    #endif

    #if LYFT_DATA == 1
        if (occlusion_counter >= 100) {
    #endif

        m_occlusion = true;
        std::cout << "there is occlusion\n";
        std::cout << "occlusion_counter = " << occlusion_counter << std::endl;
    }
    else
        m_occlusion = false;

    return dataPerRing;
}

/// split points of each ring in four regions [(1), (2), (3), (4)]
///             (3pi/2 rad)
///                  |
///             (4)  |  (1)
///  (pi rad)  (ego vehicle) (0 rad)
///             (3)  |  (2)
///                  |
///             (pi/2 rad)
/// for regions 3, 4
void CurbDetection::detection(pairIFF*& dataPerRing, VecArray& curbPoints, VecArray& testPoints, pairIIFV& results) {
    pcl::PointXYZ searchP;

    auto* region_1 = new pairFI[m_rings]; /// store the indices and angles of m_lidarPoints than belong to region (1), [angle, index]
    auto* region_2 = new pairFI[m_rings]; /// store the indices and angles of m_lidarPoints than belong to region (2), [angle, index]
    auto* region_3 = new pairFI[m_rings]; /// store the indices and angles of m_lidarPoints than belong to region (3), [angle, index]
    auto* region_4 = new pairFI[m_rings]; /// store the indices and angles of m_lidarPoints than belong to region (4), [angle, index]

    int regions_len[4] = {0, 0, 0, 0};
    int index;
    float angle;
    for (unsigned int i=0; i<m_rings; i++) {
        for (std::tuple myTuple : dataPerRing[i]) {
            index = std::get<0>(myTuple);
            angle = std::get<1>(myTuple);

            /// Search using m_projections because sorting by y values is more efficient. Search using m_lidarPoints by
            /// y value is not valid in the regions of curb because the values of y are almost identical
            searchP.x = m_lidarPoints[index].x; searchP.y = m_lidarPoints[index].y; searchP.z = m_lidarPoints[index].z;

            if(searchP.x >= m_egoPos.x && searchP.y <= m_egoPos.y) {
                region_1[i].emplace_back(angle, index);
                regions_len[0]++;
                continue;
            }
            if(searchP.x < m_egoPos.x && searchP.y < m_egoPos.y) {
                region_2[i].emplace_back(angle, index);
                regions_len[1]++;
                continue;
            }
            if(searchP.x < m_egoPos.x && searchP.y > m_egoPos.y) {
                region_3[i].emplace_back(angle, index);
                regions_len[2]++;
                continue;
            }
            if(searchP.x > m_egoPos.x && searchP.y > m_egoPos.y) {
                region_4[i].emplace_back(angle, index);
                regions_len[3]++;
                continue;
            }
        }
    }

    pointCloudBoost cloud(new pointCloud);
    SphericalNeighborhood sp;
    int id; ///  represents the number of LiDAR points that each laser line can theoretically hit on one side of the road boundary

    /// sort all regions given the value of y in order to run a middle to side search
    for (unsigned int i=0; i<m_rings; i++) {
        sort(region_1[i].begin(), region_1[i].end(), sortinrev); /// sort angle in descending order
        sort(region_2[i].begin(), region_2[i].end()); /// sort angle in increasing order
        sort(region_3[i].begin(), region_3[i].end(), sortinrev); /// sort angle in descending order
        sort(region_4[i].begin(), region_4[i].end()); /// sort angle in increasing order
    }

    std::vector<std::pair<int, float>> indices_angles;

    int regionsNumInit{1},regionsNumMax{4};
    int curbPointsNumInit{2}, curbPointsNumMax;
    int regionStep;
    regionStep=3;

    #if CARLA_DATA == 1
        curbPointsNumMax = 15; /// old
//        curbPointsNumMax = 35; /// test
    #endif

    #if LYFT_DATA == 1
//        curbPointsNumMax = 15;
        curbPointsNumMax = 40;
    #endif

    int maxCurbVec = regionsNumMax*m_rings*curbPointsNumMax; /// define the number of curb vectors candidate
    std::vector<vec> *candidates = new std::vector<vec>[maxCurbVec];
    int size, candId{0};

    /// iterate through all regions
    for (unsigned int regionId=regionsNumInit; regionId<=regionsNumMax; regionId+=regionStep) {
    /// TODO check it
        /// iterate through rings
        for (unsigned int ringId=0; ringId<m_rings; ringId++) {
            indices_angles.clear();
            if (regionId == 1 && !region_1[ringId].empty()) {
                size = region_1[ringId].size();
                /// gather lidar point for the running ring of the running region
                for (unsigned int j=0; j<size; j++) {
                    id = region_1[ringId][j].second;
                    angle = region_1[ringId][j].first;
                    indices_angles.emplace_back(id, angle);
//                    testPoints.emplace_back(m_lidarPoints[id]);
//                    testPoints.emplace_back(1.0f, 0.0f, 0.0f);
                }
            }
            if (regionId == 2 && !region_2[ringId].empty()) {
                size = region_2[ringId].size();
                /// gather lidar point for the running ring of the running region
                for (unsigned int j=0; j<size; j++) {
                    id = region_2[ringId][j].second;
                    angle = region_2[ringId][j].first;
                    indices_angles.emplace_back(id, angle);
//                    testPoints.emplace_back(m_lidarPoints[id]);
//                    testPoints.emplace_back(0.0f, 1.0f, 0.0f);
                }
            }
            if (regionId == 3 && !region_3[ringId].empty()) {
                size = region_3[ringId].size();
                /// gather lidar point for the running ring of the running region
                for (unsigned int j=0; j<size; j++) {
                    id = region_3[ringId][j].second;
                    angle = region_3[ringId][j].first;
                    indices_angles.emplace_back(id, angle);
//                    testPoints.emplace_back(m_lidarPoints[id]);
//                    testPoints.emplace_back(0.0f, 0.0f, 1.0f);
                }
            }
            if (regionId == 4 && !region_4[ringId].empty()) {
                size = region_4[ringId].size();
                /// gather lidar point for the running ring of the running region
                for (unsigned int j=0; j<size; j++) {
                    id = region_4[ringId][j].second;
                    angle = region_4[ringId][j].first;
                    indices_angles.emplace_back(id, angle);

                    if (ringId == 21) {
//                        testPoints.emplace_back(m_lidarPoints[id]);
//                        testPoints.emplace_back(1.0f, 0.0f, 0.0f);
                    }
                }
            }

            bool exist{false};
            VecArray allCurrentP;

            /// curbPointsNum must be odd in order to split equally the points before and after the candidate point
            /// iterate through different number of curb points
            for (unsigned int curbPointsNum=curbPointsNumInit; curbPointsNum<=curbPointsNumMax; curbPointsNum++) {

                /// if exist enough points
                exist = indices_angles.size() > 0; /// last
                if (exist) {
                    /// project data to bins and choose the bin with most data
                    bool found = false;
                    vec candidate;
                    varIIFV res;

                    findCandidateCurb(indices_angles, candidates, curbPointsNum, regionId, ringId, candId, testPoints, found, candidate, res);

                    /// case 3
                    if (found) {
                        results.emplace_back(res);
                        break;
                    }

                } /// if there are enough points

                exist = false;

            } /// iteration per candidate curb points

        } /// iteration per ring

    } /// iteration per region

    /// clear memory
    delete[] region_1;
    delete[] region_2;
    delete[] region_3;
    delete[] region_4;
    delete[] candidates;
}

void CurbDetection::findCandidateCurb(std::vector<std::pair<int, float>>& indices_angles, std::vector<vec>*& candidates, unsigned int& curbPointsNum, unsigned int& regionId, unsigned int& ringId, int& candId, VecArray& testPoints, bool& found, vec& candidate, varIIFV& res) {
    #if CARLA_DATA == 1
        int bins = indices_angles.size()/curbPointsNum;
    #endif
    #if LYFT_DATA == 1
//        int bins = indices_angles.size()/2*curbPointsNum;
        int bins = indices_angles.size()/curbPointsNum;
    #endif

    vec firstPoint, lastPoint;

    /// validate that in regions 3 & 4 the firstPoint.y is smaller that lastPoint.y and
    /// in regions 1 & 2 the firstPoint.y is bigger that lastPoint.y
    int firsPointInd{0}, lastPointInd;
    lastPointInd = indices_angles.size()-1;
    firstPoint = m_lidarPoints[indices_angles[firsPointInd].first];
    lastPoint = m_lidarPoints[indices_angles[lastPointInd].first];
    lastPoint.x = firstPoint.x;

    /// validate that the order of values is correct, otherwise ignore the current values
    if (regionId == 1 || regionId == 2) {
        if (firstPoint.y <= lastPoint.y)
            return;
    }
    else {
        if (firstPoint.y >= lastPoint.y)
            return;
    }

    float step = (lastPoint.y - firstPoint.y)/bins;

    std::vector<vec> *projectedPoints = new std::vector<vec>[bins];
    std::vector<float> *angles = new std::vector<float>[bins]; /// store angles of projected points
    std::unordered_map<int, int> occurrences; /// lidarToBin : map lidar id to bin id,
    /// occurrences : stores the number of points in each bin

    /// project points to bins
    vec A, B, C; /// A : firstPointToNewPoint, B : firstPointToLastPoint,C : eachProjection
    vec projection;
//    std::cout << "project points to bins...\n";
    for (auto &ind : indices_angles) {
        A = m_lidarPoints[ind.first] - firstPoint;
        B = lastPoint - firstPoint;
        C = dotProduct(A, B)/dotProduct(B, B) * B;
        projection = firstPoint + C;


        /// find in which bin exists the lidar point with this projection
        for (unsigned int j=0; j<bins; j++) {
            if (projection.y <= 0) {
                if ( ((projection.y) <= (firstPoint.y) + j*step) && ((projection.y) > (firstPoint.y) + (j+1)*step) ) {
                    /// count the number of points in each bin
                    occurrences[j] += 1;
                    projectedPoints[j].emplace_back(m_lidarPoints[ind.first]);
                    angles[j].emplace_back(ind.second);
                    break;
                } /// find the appropriate bin

                /// if there is no candidate bin, then add this point to the last bin
                if (j == bins-1) {
                    /// count the number of points in each bin
                    occurrences[j] += 1;
                    projectedPoints[j].emplace_back(m_lidarPoints[ind.first]);
                    angles[j].emplace_back(ind.second);
                }
            }
            else {
                if ( ((projection.y) >= (firstPoint.y) + j*step) && ((projection.y) < (firstPoint.y) + (j+1)*step) ) {
                    /// count the number of points in each bin
                    occurrences[j] += 1;
                    projectedPoints[j].emplace_back(m_lidarPoints[ind.first]);
                    angles[j].emplace_back(ind.second);
                    break;
                } /// find the appropriate bin

                /// if there is no candidate bin, then add this point to the last bin
                if (j == bins-1) {
                    /// count the number of points in each bin
                    occurrences[j] += 1;
                    projectedPoints[j].emplace_back(m_lidarPoints[ind.first]);
                    angles[j].emplace_back(ind.second);
                }
            }
        } /// iterate all bins

    } /// iterate all indices_angles and project them

    VecArray ringsCols;
    generateColors(bins, ringsCols);

    /// find candidate bin
    int val;
    double dif;
    bool filter_res;
    for (unsigned int j=1; j<bins; j++) {


        val = projectedPoints[j-1].size() - projectedPoints[j].size();
        if (
                projectedPoints[j].size() > 0 &&
                projectedPoints[j-1].size() > 0

            #if CARLA_DATA == 1
//                && projectedPoints[j].size() > projectedPoints[j-1].size() + curbPointsNum/3 ) { /// TODO: do i need this for carla ??????
                   ) {
            #endif

            #if LYFT_DATA == 1
//                && projectedPoints[j].size() > projectedPoints[j-1].size() + curbPointsNum/2 ) { /// old

                && projectedPoints[j].size() > projectedPoints[j-1].size() + curbPointsNum ) { /// test

//            ) {
            #endif

            /// filter points
            filter_res = filter_stage_1(projectedPoints[j], ringId);

            if (filter_res) {
                for (unsigned int w=0; w<projectedPoints[j].size()-1; w++) {

//                    dif = projectedPoints[j][w].z - projectedPoints[j][w+1].z;
//                    dif = abs(projectedPoints[j][w].z)-abs(projectedPoints[j][w+1].z);
                    dif = abs(projectedPoints[j][w].z - projectedPoints[j][w+1].z); /// (tested)

                    /// add height checking in order to ignore points that don't belong to curb
                    #if CARLA_DATA == 1
                        float t = 0.01f; /// old

//                        float t = 0.0001f; /// test

                    #endif

                    #if LYFT_DATA == 1
//                        float t = 0.0001f; /// old

                        float t = 0.01f; /// test
                    #endif

                    if (dif > t) {
                        candidate = projectedPoints[j][w];
                        res = varIIFV(regionId, ringId, angles[j][w], candidate);
                        found = true;
                        candId++;
                        j=bins;
                        break;
                    }
                }
            } /// filter_stage_1
        }
    } /// iterate through bins

    delete[] projectedPoints;
}

/// predict more curb points
void CurbDetection::predict(pairIIFV& data) {}

/// Function for calculating variance
float CurbDetection::variance(std::vector<float> values) {
    /// Compute mean (average of elements)
    float mean = getMean(values);

    /// Compute sum squared differences with mean.
    double sqDiff = 0;
    for (float value : values)
        sqDiff += (value - mean) * (value - mean);
    return sqDiff / values.size();
}

Eigen::VectorXd CurbDetection::polynomial_ransac(VecArray& points, double& threshold_old, bool& success) {
    int degree{2};

    /// curv p : y = p2x^2 + p1x + p0

    int iterations = 10000;
    VecArray randomPoints, inliers, allInliers;
    int lowerBound{0}, upperBound = points.size();
    double dist;
    vec randId, rand1, rand2, rand3;
    int tmp_highest_num_of_inliers = (60*points.size())/100;

    double t;
    for (double threshold=4.0f ; threshold <=4.0f; threshold+=0.1f) {
        for (unsigned int i=0; i<iterations; i++) {
            inliers.clear();

            randomPoints.clear();
            /// select degree + 1 random three points
            for (unsigned int j=0; j<degree+1; j++) {
                randId = points[(rand() % upperBound) + lowerBound];
                randomPoints.emplace_back(randId);
            }
            while (sameVec(randomPoints[0],randomPoints[1]) || sameVec(randomPoints[0],randomPoints[2]) || sameVec(randomPoints[1],randomPoints[2])) {
                randomPoints.clear();
                for (unsigned int j=0; j<degree+1; j++) {
                    randId = points[(rand() % upperBound) + lowerBound];
                    randomPoints.emplace_back(randId);
                }
            }

            /// construct the polynomial curve of degree 2 through these three points
            /// given the tree points (x0,y0), (x1,y1) and (x2,y2) solve the below system
            /*
            --  --      --         --       --  --
            | y0 |      | x0^2 x0 1 |       | p2 |
            | y1 |  =   | x1^2 x1 1 |   *   | p1 |
            | y2 |      | x2^2 x2 1 |       | p0 |
            --  --      --         --       --  --
               Y    =         X         *      p
            */
            Eigen::MatrixXd X(degree+1,degree+1);
            Eigen::VectorXd Y(degree+1,1), p(degree+1,1);
            for (unsigned int j=0; j<degree+1; j++) {
                X(j,0) = pow(randomPoints[j].x, 2);
                X(j,1) = randomPoints[j].x;
                X(j,2) = 1;
                Y(j) = randomPoints[j].y;
            }

            /// find curve
            p = X.inverse()*Y;

            /// Test all other points against the model p
            for (auto &point : points) {
                if ( sameVec(point, randomPoints[0]) || sameVec(point, randomPoints[1]) || sameVec(point, randomPoints[2]) ) {
                    continue;
                }
                else {
                    /// Compute the distance from this curve 𝒑 for each other point
                    /// I need to minimize the distance function for each point and find the minimum distance from each
                    /// point (x0,y0) to the curve p
                    /// distance function : l(x) = sqrt( (x-x0)^2 + (p2x^2 + p1x + p0 - y0)^2 )

                    /// Solve derivative of l(x) to x :
                    ///     dl(x)
                    ///     ----- = 0
                    ///      dx

                    /// the result after the derivative is :
                    /// x^3 + 3p1x^2 + (1+2p2(p0-y0)+p1^2)x + (p1(-y0+p0)-x0) = 0
                    ///       -------   -------------------   ---------------
                    ///          2              2p2                 2p2

                    double x[degree+1], a, b, c;

                    if (degree == 2) {
                        a = 3*p(1)/2;
                        b = ((1+2*p(0)*(p(2)-point.y)+p(1)*p(1)))/2*p(0);
                        c = (p(1)*(-point.y+p(2))-point.x)/(2*p(0));

                        if (isinf(a) || isinf(b) || isinf(c))
                            continue;

                        /// solve cubic equation x^3 + a*x^2 + b*x + c = 0
                        int res = SolveP3(x, a, b, c);

                        /// TODO check real root...
                        /// 1 real root (The real root defines the -coordinate of the nearest point on the curve)
//                        if (res != 1) {
                        /// check if minimum distance from point to curve (point (x[0], curve(x[0]))) is less that threshold
                        dist = sqrt(pow(x[0] - point.x, 2) + pow( p(0)*x[0]*x[0] + p(1)*x[0] + p(2) - point.y, 2));
                        if (dist < threshold)
                            inliers.emplace_back(point);
//                        } /// 1 real root
                    } /// only for degree == 2

                } /// check i points is different from the above random three points

            } /// Test all other points against the model p

            /// If the actual number of inliers is greater than the temporary highest number of inliers,
            /// than the model and the temporary number of inliers is saved
            if (inliers.size() >= tmp_highest_num_of_inliers) {
                for (auto &n : inliers) {
                    if (existInVec(allInliers, n)) {}
                    else {
                        allInliers.emplace_back(n);
                    }
                }
                tmp_highest_num_of_inliers = inliers.size();
            }
        } /// iterations
        t = threshold;
    } /// for each threshold

    std::cout << "threshold = " << t << std::endl;
    std::cout << "all inliers are = " << allInliers.size() << std::endl;
    std::cout << "lineFitting ...\n" << std::endl;

    Eigen::VectorXd final(4+1,1);

    if (allInliers.size() == 0) {
        success = false;
        return final;
    }

    std::vector<float> testCoeffs, allX, allY;
    for (auto &n : allInliers) {
        allX.emplace_back(n.x);
        allY.emplace_back(n.y);
    }

    lineFitting(allX, allY, 4, testCoeffs); /// TODO change name to polynomial fitting
    for (unsigned int j=0; j<4+1; j++) {
        final(j) = testCoeffs[j];
        std::cout << final(j) << std::endl;
    }

    return final;
}

Eigen::VectorXd CurbDetection::linear_ransac(VecArray& points, double& threshold, bool& success) {
    pcl::SampleConsensusModel<pcl::PointXYZ>::PointCloudPtr cloud(new pcl::SampleConsensusModel<pcl::PointXYZ>::PointCloud);
    pcl::SampleConsensusModel<pcl::PointXYZ>::PointCloudPtr final(new pcl::SampleConsensusModel<pcl::PointXYZ>::PointCloud);

    int indPoint{0};
    std::vector<int> samples;
    std::vector<double> distances;
    std::vector<int> inliers;
    Eigen::VectorXf optimized_coefficients;

    /// resize space for point cloud
    cloud->points.clear();
    cloud->width = points.size(); cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);
    indPoint = 0;
    for (auto & point : cloud->points) {
        point.x = points[indPoint].x;
        point.y = points[indPoint].y;
        point.z = points[indPoint].z;
        indPoint++;
    }

    pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr
            model_l(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_l);
//    ransac.setDistanceThreshold (.1); /// old
    ransac.setDistanceThreshold (1.0); /// test
    ransac.computeModel();
    ransac.getInliers(inliers);
    pcl::copyPointCloud (*cloud, inliers, *final);

    Eigen::VectorXf model_coef;
    /// The six coefficients of the line are given by a point on the line and the direction of the line as:
    /// [point_on_line.x point_on_line.y point_on_line.z line_direction.x line_direction.y line_direction.z]
    ransac.getModelCoefficients(model_coef);

    points.clear();
    for (auto & point : final->points)
        points.emplace_back(vec(point.x, point.y, point.z));


    std::vector<float> testCoeffs, allX, allY;
    for (auto &n : points) {
        allX.emplace_back(n.x);
        allY.emplace_back(n.y);
    }

    lineFitting(allX, allY, 1, testCoeffs); /// TODO change name to polynomial fitting
    Eigen::VectorXd coeff(2,1);
    for (unsigned int j=0; j<2; j++)
        coeff(j) = testCoeffs[j];

    return coeff;
}

bool CurbDetection::isCurved(VecArray& points, float& t) {
    Eigen::VectorXd roadUp, roadBelow;
    std::vector<float> y_values, x_values;
    float y_var, x_var;
    for (auto &p : points) {
        y_values.emplace_back(p.y);
        x_values.emplace_back(p.x);
    }
    y_var = variance(y_values);
    x_var = variance(x_values);

    std::cout << "y var for curb = " << y_var << std::endl;
    std::cout << "x var for curb = " << x_var << std::endl;

    return x_var > t && y_var > t; /// true -> curved line
}

/// flag = True => curved line
/// flag = False => straight line
Eigen::VectorXd CurbDetection::getRoadEquation(bool flag, VecArray& points, bool& success) {
    Eigen::VectorXd coeffs;
    double tmp{0.1};

    /// curved line up
    if (flag)
        coeffs = polynomial_ransac(points, tmp, success);
    /// straight line up
    else
        coeffs = linear_ransac(points, tmp, success);
    return coeffs;
}

void CurbDetection::getRoadPoints(bool& isCurvedUp, bool& isCurvedBelow, Eigen::VectorXd roadUp, Eigen::VectorXd roadBelow, VecArray& curbUp, VecArray& curbBelow, VecArray& allPoints, VecArray& roadPointsVector) {
    float val_a, val_b;
    bool insideUp{false}, insideBelow{false};

    /// find the edge points of each road line
    vec roadUpMin = getPointWithMinX(curbUp);
    vec roadUpMax = getPointWithMaxX(curbUp);
    vec roadBelowMin = getPointWithMinX(curbBelow);
    vec roadBelowMax = getPointWithMaxX(curbBelow);

    std::vector<float> curbX, curbY;
    for (auto &p : curbUp) {
        curbY.emplace_back(p.y);
        curbX.emplace_back(p.x);
    }
    for (auto &p : curbBelow) {
        curbY.emplace_back(p.y);
        curbX.emplace_back(p.x);
    }
    float minCurbY = getMin(curbY);
    float maxCurbY = getMax(curbY);
    float minCurbX = getMin(curbX);
    float maxCurbX = getMax(curbX);

    if (!isCurvedUp && !isCurvedBelow) {
//        std::cout << "case 1\n";
        for (auto & i : allPoints) {
            val_a = roadUp[1]*i.x + roadUp[0];
            val_b = roadBelow[1]*i.x + roadBelow[0];

            insideUp = i.x >= roadUpMin.x && i.x <= roadUpMax.x;
            insideBelow = i.x >= roadBelowMin.x && i.x <= roadBelowMax.x;

            # if CARLA_DATA == 1
                if ( (((i.y >= val_a) && (i.y < val_b)) || ((i.y < val_a) && (i.y >= val_b))) && (insideUp || insideBelow)
                     && (i.y >= minCurbY && i.y <= maxCurbY) ) {
            #endif

            #if LYFT_DATA == 1
                if ( (((i.y >= val_a) && (i.y < val_b)) || ((i.y < val_a) && (i.y >= val_b)))
                     && (i.y >= minCurbY && i.y <= maxCurbY) && (i.x >= minCurbX && i.x <= maxCurbX)  ) {
            #endif

                    roadPointsVector.emplace_back(i);
            }
        }
    }
    if (!isCurvedUp && isCurvedBelow) {
//        std::cout << "case 2\n";/
        for (auto & i : allPoints) {
            val_a = roadUp[1]*i.x + roadUp[0];
            val_b = roadBelow[4]*pow(i.x,4) + roadBelow[3]*pow(i.x,3) + roadBelow[2]*pow(i.x,2) + roadBelow[1]*i.x + roadBelow[0];
            insideUp = i.x >= roadUpMin.x && i.x <= roadUpMax.x;
            insideBelow = i.x >= roadBelowMin.x && i.x <= roadBelowMax.x;

            #if CARLA_DATA == 1
                if ( (((i.y >= val_a) && (i.y < val_b)) || ((i.y < val_a) && (i.y >= val_b))) && (insideUp || insideBelow)
                     && (i.y >= minCurbY && i.y <= maxCurbY) ) {
            #endif

            #if LYFT_DATA == 1
                if ( (((i.y >= val_a) && (i.y < val_b)) || ((i.y < val_a) && (i.y >= val_b)))
                     && (i.y >= minCurbY && i.y <= maxCurbY) && (i.x >= minCurbX && i.x <= maxCurbX)  ) {
            #endif

                    roadPointsVector.emplace_back(i);
            }
        }
    }
    if (isCurvedUp && !isCurvedBelow) {
//        std::cout << "case 3\n";
        for (auto & i : allPoints) {
            val_a = roadUp[4]*pow(i.x,4) + roadUp[3]*pow(i.x,3) + roadUp[2]*pow(i.x,2) + roadUp[1]*i.x + roadUp[0];
            val_b = roadBelow[1]*i.x + roadBelow[0];
            insideUp = i.x >= roadUpMin.x && i.x <= roadUpMax.x;
            insideBelow = i.x >= roadBelowMin.x && i.x <= roadBelowMax.x;
#if CARLA_DATA == 1
            if ( (((i.y >= val_a) && (i.y < val_b)) || ((i.y < val_a) && (i.y >= val_b))) && (insideUp || insideBelow)
                 && (i.y >= minCurbY && i.y <= maxCurbY) ){
#endif
#if LYFT_DATA == 1
            if ( (((i.y >= val_a) && (i.y < val_b)) || ((i.y < val_a) && (i.y >= val_b)))
                 && (i.y >= minCurbY && i.y <= maxCurbY) && (i.x >= minCurbX && i.x <= maxCurbX)  ) {
#endif

                roadPointsVector.emplace_back(i);
            }
        }
    }
    if (isCurvedUp && isCurvedBelow) {
//        std::cout << "case 4\n";
        for (auto & i : allPoints) {
            val_a = roadUp[4]*pow(i.x,4) + roadUp[3]*pow(i.x,3) + roadUp[2]*pow(i.x,2) + roadUp[1]*i.x + roadUp[0];
            val_b = roadBelow[4]*pow(i.x,4) + roadBelow[3]*pow(i.x,3) + roadBelow[2]*pow(i.x,2) + roadBelow[1]*i.x + roadBelow[0];
            insideUp = i.x >= roadUpMin.x && i.x <= roadUpMax.x;
            insideBelow = i.x >= roadBelowMin.x && i.x <= roadBelowMax.x;

            #if CARLA_DATA == 1
            /// old
            if ( (((i.y >= val_a) && (i.y < val_b)) || ((i.y < val_a) && (i.y >= val_b)))
            && (insideUp || insideBelow)
                 && (i.y >= minCurbY && i.y <= maxCurbY)
                 ) {

            #endif
#if LYFT_DATA == 1
                if ( (((i.y >= val_a) && (i.y < val_b)) || ((i.y < val_a) && (i.y >= val_b)))
                     && (i.y >= minCurbY && i.y <= maxCurbY) && (i.x >= minCurbX && i.x <= maxCurbX)  ) {
#endif

                roadPointsVector.emplace_back(i);
            }
        }
    }
}

vec CurbDetection::getPointWithMinX(const VecArray& values) {
    float min = values[0].x;
    int id{0}, c{0};
    for (auto &v : values) {
        if (v.x < min) {
            id = c;
            min = v.x;
        }
        c++;
    }
    return values[id];
}

vec CurbDetection::getPointWithMaxX(const VecArray& values) {
    float max = values[0].x;
    int id{0}, c{0};
    for (auto &v : values) {
        if (v.x > max) {
            id = c;
            max = v.x;
        }
        c++;
    }
    return values[id];
}

bool CurbDetection::filter_stage_1(std::vector<vec>& points, int ringId) {
    pointCloudBoost cloud(new pointCloud);

    /// cluster noGround points
    pcl::PCDWriter writer;
    /// resize space for point cloud
    cloud->width = points.size(); cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);
    int id{0};
    for (auto & point : cloud->points) {
        point.x = points[id].x; point.y = points[id].y; point.z = points[id].z;
        id++;
    }

    /// Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    /// TODO
    /// set tolerance as parameter of variance of curb points, biggest variance => biggest tolerance

//    ec.setClusterTolerance (0.1); /// old
    ec.setClusterTolerance (0.5); /// last tested
//    ec.setClusterTolerance (0.2);
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(200);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    return cluster_indices.size() <= 1;
}

bool CurbDetection::lineFitting(std::vector<float>& x, std::vector<float>& y, const int& order, std::vector<float>& coeffs)
{
    /// The size of xValues and yValues should be same
    if (x.size() != y.size()) {
        throw std::runtime_error("The size of x & y arrays are different");
        return false;
    }
    /// The size of xValues and yValues cannot be 0, should not happen
    if (x.size() == 0 || y.size() == 0) {
        throw std::runtime_error("The size of x or y arrays is 0");
        return false;
    }

    size_t N = x.size();
//    int n = 2;
    int n = order;
    int np1 = n + 1;
    int np2 = n + 2;
    int tnp1 = 2 * n + 1;
    double tmp;

    std::vector<double> X(tnp1);
    for (int i = 0; i < tnp1; ++i) {
        X[i] = 0;
        for (int j = 0; j < N; ++j)
            X[i] += (double)pow(x[j], i);
    }

    /// a = vector to store final coefficients.
    std::vector<double> a(np1);

    /// B = normal augmented matrix that stores the equations.
    std::vector<std::vector<double> > B(np1, std::vector<double>(np2, 0));

    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= n; ++j)
            B[i][j] = X[i + j];

    /// Y = vector to store values of sigma(xi^n * yi)
    std::vector<double> Y(np1);
    for (int i = 0; i < np1; ++i) {
        Y[i] = (double)0;
        for (int j = 0; j < N; ++j) {
            Y[i] += (double)pow(x[j], i) * y[j];
        }
    }

    /// Load values of Y as last column of B
    for (int i = 0; i <= n; ++i)
        B[i][np1] = Y[i];

    n += 1;
    int nm1 = n - 1;

    /// Pivotisation of the B matrix.
    for (int i = 0; i < n; ++i)
        for (int k = i + 1; k < n; ++k)
            if (B[i][i] < B[k][i])
                for (int j = 0; j <= n; ++j) {
                    tmp = B[i][j];
                    B[i][j] = B[k][j];
                    B[k][j] = tmp;
                }

    /// Performs the Gaussian elimination.
    /// (1) Make all elements below the pivot equals to zero
    ///     or eliminate the variable.
    for (int i = 0; i < nm1; ++i)
        for (int k = i + 1; k < n; ++k) {
            double t = B[k][i] / B[i][i];
            for (int j = 0; j <= n; ++j)
                B[k][j] -= t * B[i][j];         /// (1)
        }

    /// Back substitution.
    /// (1) Set the variable as the rhs of last equation
    /// (2) Subtract all lhs values except the target coefficient.
    /// (3) Divide rhs by coefficient of variable being calculated.
    for (int i = nm1; i >= 0; --i) {
        a[i] = B[i][n];                   /// (1)
        for (int j = 0; j < n; ++j)
            if (j != i)
                a[i] -= B[i][j] * a[j];       /// (2)
        a[i] /= B[i][i];                  /// (3)
    }

    coeffs.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        coeffs[i] = a[i];

    return true;
}

SphericalCoordinates CurbDetection::convertToSpherical(const vec &point) {
    SphericalCoordinates sc;
    sc.r = float(sqrt(float(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2))));

    /// Inspired by https://stackoverflow.com/questions/283406/what-is-the-difference-between-atan-and-atan2-in-c/12011762#12011762
    sc.th = float(atan2(point.y, point.x) * 180.0f / M_PI); /// convert radian to degree

    /// the resulting th angles from atan2 include some negative values, so we modify them in order to be in the range [0, 360]
    if(sc.th < 0.0f)
        sc.th += 360.0f;

//    std::cout << "th = " << sc.th << std::endl;

    sc.f = float(atan2(sqrt(point.x*point.x + point.y*point.y), point.z) * 180.0f / M_PI); /// convert radian to degree
    return sc;
}

void CurbDetection::generateColors(const int& num, VecArray& cols) {
    float randR, randG, randB;
    for (unsigned int i=0; i<num; i++) {
        randR = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        randG = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        randB = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        cols.emplace_back(randR, randG, randB);
    }
}

void CurbDetection::setProjectionPoints(const VecArray& points) {
    m_projections = points;
}

float CurbDetection::diff(const vec&p1, const vec&p2) {

    return sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2) + pow(p1.z-p2.z, 2));
}

/// compute the angle between three consecutive points p1, p2 and p3
///            p3
///            -------
///            -
///    angle ( -
///     ------- p2
///   p1
/// TODO check this funciton
float CurbDetection::getAngle(const vec& p1, const vec& p2, const vec& p3) {
    /// related to https://knowledge.autodesk.com/search-result/caas/CloudHelp/cloudhelp/2015/ENU/MAXScript-Help/files/GUID-8FB870E5-823A-42FE-9A8C-70D25FF3B92C-htm.html
    vec v1, v2;
    v1 = p3-p2;
    v2 = p1-p2;
    return v1.AngleBetween(v2)* 180.0f / M_PI;
//    v1.Normalize();
}

double CurbDetection::dotProduct(const vec& v1, const vec& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

void CurbDetection::pca(VecArray& points, Eigen::Matrix3f& eigen_vectors, Eigen::Vector3f& eigen_values, VecArray& testPoints, vec currentP) {
    pcl::PointCloud<pcl::PointXYZ> neighbors_cloud;

    /// fill cloud with the points of previous ring and find the nearest neighbor of currentP
    neighbors_cloud.width = points.size(); neighbors_cloud.height = 1;
    neighbors_cloud.points.resize (neighbors_cloud.width * neighbors_cloud.height);
    for (unsigned int m=0; m<neighbors_cloud.points.size(); m++) {
        neighbors_cloud.points[m].x = points[m].x;
        neighbors_cloud.points[m].y = points[m].y;
        neighbors_cloud.points[m].z = points[m].z;
    }

    /// Placeholder for the 3x3 covariance matrix at each surface patch
    Eigen::Matrix3f covariance_matrix;
    /// 16-bytes aligned placeholder for the XYZ centroid of a surface patch
    Eigen::Vector4f xyz_centroid;
    /// Estimate the XYZ centroid
    pcl::compute3DCentroid(neighbors_cloud, xyz_centroid);
    /// Compute the 3x3 covariance matrix
    pcl::computeCovarianceMatrix(neighbors_cloud, xyz_centroid, covariance_matrix);

    /// Extract the eigenvalues and eigenvectors
    pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);
}

bool CurbDetection::existInVec(VecArray values, vec value) {
    for (auto &val : values) {
        if ( (val.x == value.x) && (val.y == value.y) && (val.z == value.z))
            return true;
    }
    return  false;
}

int CurbDetection::getBins() {
    return m_bins;
}

bool CurbDetection::sameVec(vec p1, vec p2) {
    if ( (p1.x == p2.x) && (p1.y == p2.y) && (p1.z == p2.z) )
        return true;
    return false;
}

template <typename T>
T CurbDetection::getMin(const std::vector<T>& values) {
    T min = values[0];
    for (auto &v : values)
        if (v < min)
            min = v;
    return min;
}

template <typename T>
T CurbDetection::getMax(const std::vector<T>& values) {
    T max = values[0];
    for (auto &v : values)
        if (v > max)
            max = v;
    return max;
}

template <typename T>
T CurbDetection::getMean(const std::vector<T>& values) {
    T res{0.0f};
    for (auto &v : values)
        res += v;
    res /= values.size();
    return res;
}