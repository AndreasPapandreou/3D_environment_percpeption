#include "Classifier.h"

Classifier::Classifier(std::string classifierPath) {
    m_classifierPath = classifierPath;
}

/// classify each point separately
void Classifier::run(VecArray& points, std::vector<int>& categories) {
    std::stringstream trainingData;
    pointCloudBoost cloud(new pointCloud);
    SphericalNeighborhood sNeighbors;
    vec center;

    /// TODO must optimize these parameters => (initialR, smallestRadius, ratio, scalesNum)
    /// setting for testing
    float initialR = 7.0f;
    float smallestRadius = 0.5f;
    float ratio = 1.1f;
    unsigned int scalesNum = 20;

    /// classify each cluster
    trainingData.str("");
    trainingData << "type eigenvaluesSum omnivariance eigenentropy linearity planarity sphericity curvatureChange "
                    "verticalityFirstEigenvectorAxisZ verticalityThirdEigenvectorAxisZ absoluteMomentFirstOrderE1 "
                    "absoluteMomentFirstOrderE2 absoluteMomentFirstOrderE3 absoluteMomentSecondOrderE1 "
                    "absoluteMomentSecondOrderE2 absoluteMomentSecondOrderE3 verticalMomentFirstOrder "
                    "verticalMomentSecondOrder pointsNumber\n";

    cloud->clear();
    /// resize space for point cloud
    cloud->width = points.size()*3; cloud->height = 1;
    cloud->points.resize (cloud->width * cloud->height);

    /// fill point cloud with data and compute center of cluster
    int indPoint{0};
    center.x = 0.0f; center.y = 0.0f; center.z = 0.0f;
    for (auto & point : cloud->points) {
        point.x = points[indPoint].x;
        point.y = points[indPoint].y;
        point.z = points[indPoint].z;
        indPoint++;
    }

    sNeighbors.init(initialR, scalesNum, smallestRadius, ratio, cloud);

    int id{0};
    /// classify each point
    for (auto &p : points) {
        /// extract geometrical features
        int index{-1};
        sNeighbors.setCenter(p);
        sNeighbors.setCloud(cloud);

        VecArray tmp;
        sNeighbors.run(trainingData, index, tmp);

        /// predict the class
        int res = predict(trainingData);
        categories.emplace_back(res);

        std::cout << id << " / " << points.size() << std::endl;
        id++;

    }
}


/// run classifier for one object only
int Classifier::run(VecArray& points) {
    std::stringstream trainingData;
    pointCloudBoost cloud(new pointCloud);
    SphericalNeighborhood sNeighbors;
    vec center;

    /// TODO must optimize these parameters => (initialR, smallestRadius, ratio, scalesNum)
    /// setting for walkers
    float initialR = 7.0f;
    float smallestRadius = 0.5f;
    float ratio = 1.1f;
    unsigned int scalesNum = 20;

    std::vector<float> cluster, allX, allY, allZ;
    float minX, maxX, minY, maxY, minZ, maxZ;

    for (auto &p : points) {
        cluster.emplace_back(p.x);
        cluster.emplace_back(p.y);
        cluster.emplace_back(p.z);
    }

    /// classify each cluster
    trainingData.str("");
    trainingData << "type eigenvaluesSum omnivariance eigenentropy linearity planarity sphericity curvatureChange "
                    "verticalityFirstEigenvectorAxisZ verticalityThirdEigenvectorAxisZ absoluteMomentFirstOrderE1 "
                    "absoluteMomentFirstOrderE2 absoluteMomentFirstOrderE3 absoluteMomentSecondOrderE1 "
                    "absoluteMomentSecondOrderE2 absoluteMomentSecondOrderE3 verticalMomentFirstOrder "
                    "verticalMomentSecondOrder pointsNumber\n";

    cloud->clear();
    /// resize space for point cloud
    cloud->width = cluster.size()/3; cloud->height = 1;
    cloud->points.resize (cloud->width * cloud->height);

    /// fill point cloud with data and compute center of cluster
    int indPoint{0};
    center.x = 0.0f; center.y = 0.0f; center.z = 0.0f;
    for (auto & point : cloud->points) {
        point.x = cluster[indPoint];
        point.y = cluster[indPoint+1];
        point.z = cluster[indPoint+2];
        indPoint+=3;

        allX.emplace_back(point.x);
        allY.emplace_back(point.y);
        allZ.emplace_back(point.z);
    }

    minX = getMin(allX);
    maxX = getMax(allX);

    minY = getMin(allY);
    maxY = getMax(allY);

    minZ = getMin(allZ);
    maxZ = getMax(allZ);

    center.x = (minX+maxX)/2;
    center.y = (minY+maxY)/2;
    center.z = (minZ+maxZ)/2;

    sNeighbors.init(initialR, scalesNum, smallestRadius, ratio, cloud);

    /// extract geometrical features
    int index{-1};
    sNeighbors.setCenter(center);
    sNeighbors.setCloud(cloud);

    VecArray tmp;
    sNeighbors.run(trainingData, index, tmp);

    /// predict the class
    int res = predict(trainingData);
    return res;
}

void Classifier::run(pcl::PointCloud<PointTypeIO>::Ptr& inner_cloud, pcl::IndicesClustersPtr& clusters, std::vector<int>& categories) {
    std::stringstream trainingData;
    pointCloudBoost cloud(new pointCloud);
    SphericalNeighborhood sNeighbors;
    vec center;

    /// TODO must optimize these parameters => (initialR, smallestRadius, ratio, scalesNum)
    /// setting for walkers
    float initialR = 7.0f;
    float smallestRadius = 0.5f;
//    float smallestRadius = 0.3f;
    float ratio = 1.1f;
//    unsigned int scalesNum = 15;
    unsigned int scalesNum = 20;

    std::vector<float> cluster, allX, allY, allZ;
    float minX, maxX, minY, maxY, minZ, maxZ;

    /// iterate through all clusters
    for (int clus = 0; clus < clusters->size (); ++clus) {
        allX.clear();
        allY.clear();
        allZ.clear();
        cluster.clear();
        for (int index = 0; index < (*clusters)[clus].indices.size (); ++index) {
            cluster.emplace_back(inner_cloud->points[(*clusters)[clus].indices[index]].x);
            cluster.emplace_back(inner_cloud->points[(*clusters)[clus].indices[index]].y);
            cluster.emplace_back(inner_cloud->points[(*clusters)[clus].indices[index]].z);
        }

        /// classify each cluster
        trainingData.str("");
        trainingData << "type eigenvaluesSum omnivariance eigenentropy linearity planarity sphericity curvatureChange "
                        "verticalityFirstEigenvectorAxisZ verticalityThirdEigenvectorAxisZ absoluteMomentFirstOrderE1 "
                        "absoluteMomentFirstOrderE2 absoluteMomentFirstOrderE3 absoluteMomentSecondOrderE1 "
                        "absoluteMomentSecondOrderE2 absoluteMomentSecondOrderE3 verticalMomentFirstOrder "
                        "verticalMomentSecondOrder pointsNumber\n";

        cloud->clear();
        /// resize space for point cloud
        cloud->width = cluster.size()/3; cloud->height = 1;
        cloud->points.resize (cloud->width * cloud->height);

        /// fill point cloud with data and compute center of cluster
        int indPoint{0};
        center.x = 0.0f; center.y = 0.0f; center.z = 0.0f;
        for (auto & point : cloud->points) {
            point.x = cluster[indPoint];
            point.y = cluster[indPoint+1];
            point.z = cluster[indPoint+2];
            indPoint+=3;

            allX.emplace_back(point.x);
            allY.emplace_back(point.y);
            allZ.emplace_back(point.z);
        }

        minX = getMin(allX);
        maxX = getMax(allX);

        minY = getMin(allY);
        maxY = getMax(allY);

        minZ = getMin(allZ);
        maxZ = getMax(allZ);

        center.x = (minX+maxX)/2;
        center.y = (minY+maxY)/2;
        center.z = (minZ+maxZ)/2;

        sNeighbors.init(initialR, scalesNum, smallestRadius, ratio, cloud);

        /// extract geometrical features
        int index{-1};
        sNeighbors.setCenter(center);
        sNeighbors.setCloud(cloud);

        VecArray tmp;
        sNeighbors.run(trainingData, index, tmp);

        /// predict the class
        int res = predict(trainingData);
        categories.emplace_back(res);
    }
}

int Classifier::predict(std::stringstream& trainingData) {
    Helpers hp;
    std::vector<int> eachPrediction;

    std::ofstream outMyfile;
    outMyfile.open("data/featureForPrediction.dat", std::ios::out | std::ios::trunc);
    outMyfile << trainingData.str();
    outMyfile.close();

    std::string pwd = hp.getCurrentDir();
    std::string cmd{"cd " + m_classifierPath + " && ./ranger --file " + pwd + "/data/featureForPrediction.dat --predict " + m_classifierPath + "ranger_out.forest"};
    int n = cmd.length();
    char char_array[n + 1]; /// declaring character array
    strcpy(char_array, cmd.c_str()); /// copying the contents of the string to char array
    system(char_array);

    std::string line;
    std::ifstream inMyfile (m_classifierPath + "/ranger_out.prediction");
    if (inMyfile.is_open())
    {
        while (getline(inMyfile,line)) {
            if (line.compare("Predictions: ") != 0)
                eachPrediction.emplace_back(std::stoi(line));
        }
        inMyfile.close();
    }

    /// get the value with the most frequency
    /// Insert all elements in hash.
    std::unordered_map<int, int> hash;
    for (int m : eachPrediction)
        hash[m]++;

    /// find the max frequency
    int max_count = 0, res = -1;
    for (auto n : hash) {
        if (max_count < n.second) {
            res = n.first;
            max_count = n.second;
        }
    }

    if (res != 0 && res != 1 && res != 2)
        res = -1;
    return res;
}

template <typename T>
T Classifier::getMin(const std::vector<T>& values) {
    T min = values[0];
    for (auto &v : values)
        if (v < min)
            min = v;
    return min;
}

template <typename T>
T Classifier::getMax(const std::vector<T>& values) {
    T max = values[0];
    for (auto &v : values)
        if (v > max)
            max = v;
    return max;
}
