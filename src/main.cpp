#define GLM_ENABLE_EXPERIMENTAL

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

#include "Camera.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include <pcl/point_types.h>
#include <pcl/console/time.h>
#include "pcl/ml/kmeans.h"
#include <bits/stdc++.h>
#include <map>
#include <fstream>
#include <chrono>
/// prediction model
#include "SystemModel.h"
#include "PositionMeasurementModel.h"
#include <kalman/ExtendedKalmanFilter.hpp>
#include <FreeImage.h>
#include "stb-master/stb_image_write.h"

#include "Renderer.h"
#include "VertexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"
#include "VertexBufferLayout.h"
#include "Camera.h"
#include "LyftDatasetHandler.h"
#include "DataConversions.h"
#include "SphericalNeighborhood.h"
#include "Segmentation.h"
#include "Helpers.h"
#include "Classifier.h"
#include "CurbDetection.h"
#include "Clustering.h"

#define TRAINING 0
#define TESTING 0
#define SIMPLE_RUN 1
#define SIMPLE_RUN_WITH_CLASSIFIER_ONLY 0
#define RECORD 0
#define CARLA_DATA 1
#define LYFT_DATA 0

#define SPHERICAL_NEIGHBORHOOD 0
#define DRAW_BOXES 0
#define DRAW_PLANE 0
#define DRAW_OBJECT 0
#define DRAW_RINGS 0 /// if draw rings then comment draw original points
#define DRAW_CURB 0
#define DRAW_ROAD 0
#define DRAW_TEST 0
#define DRAW_GROUND_V1 0
#define DRAW_NON_GROUND_V1 0
#define DRAW_SEGMENTS_V1 0
#define DRAW_SEGMENTS_V2 0
#define DRAW_ROAD_LINE 0
#define DRAW_ORIGINAL_POINTS 1
#define DRAW_PROJECTIONS 0
#define DRAW_NEIGHBORS 0
#define RING_OUTLIERS 0
#define ROAD_POINTS_VECTOR 0
#define DRAW_ORIENTATION_VEC 0
#define DRAW_VEHICLES 0
#define DRAW_WALKERS 0
#define DRAW_ROAD_CARLA 0
#define DRAW_OBJECTS_ABOVE_ROAD 0

typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;
typedef pcl::PointXYZ Point;
using namespace std::chrono;
namespace fs = std::filesystem;

using namespace KalmanExamples;
typedef float T;
typedef Robot1::State<T> State;
typedef Robot1::Control<T> Control;
typedef Robot1::SystemModel<T> SystemModel;
typedef Robot1::PositionMeasurement<T> PositionMeasurement;

int init();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
std::string setDataForTraining(std::string& pathTolyftData, LyftDatasetHandler& dh, std::vector<Annotation>& anns, vec& trainCenter, VecArray& debug);
void setCarlaDataForTraining(VecArray& points, Box3D& box, VecArray& debug, std::stringstream& trainingData);
void saveImage(const char* filepath, GLFWwindow* w);
float diff(const vec&p1, const vec&p2);
void showImages(std::string title, Sample& sample);
void showImages(std::string title, int nArgs, std::vector<std::string>& images);
bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance);
template <typename T>
T getMin(const std::vector<T>& values);
template <typename T>
T getMax(const std::vector<T>& values);
void readData(std::string &pathToData, VecArray& points);
void readMetaData(std::string &pathToData, vec& camera_pos, vec& lidar_pos, VecArray& vehicle_pos, VecArray& extrinsics, std::vector<bool>& vehicle_moving);
void estimateRoadLines(Eigen::VectorXd &roadUp, Eigen::VectorXd& roadBelow, VecArray& curbUp, VecArray& curbBelow, VecArray &curbPoints, VecArray &roadPointsVector, VecArray &totalGroundV1, VecArray &totalNoGroundV1, vec &ego_location, VecArray& testup, VecArray& testbelow, int draw_road_line);
void write_to_bin(VecArray& points, std::string& filename, bool flag);

/// settings
const unsigned int SCR_WIDTH = 2048;
const unsigned int SCR_HEIGHT = 1024;
GLFWwindow* window;

/// create hash map with all categories
std::unordered_map<std::string, int> catToInt;

/// initialize camera object
glm::vec3 temporalEgoPos(0.0f, 0.0f, 0.0f);
glm::vec3 egoPos(0.0f, 0.0f, 0.0f);
Camera camera(temporalEgoPos);
Renderer renderer;
int dimensions{3};

/// define paths for lyft data
std::string jsonLidarSmall = "/data/lyftMetadataSmall.json";
std::string jsonLidarMedium = "/data/lyftMetadataMedium.json";
std::string jsonLidarBig = "/data/lyftMetadataBig.json";

/// define paths for nuscenes data
std::string pathTolyftData;
std::string classifierPath;
int frameStart, frameEnd;
int frames = frameEnd - frameStart;
int iter;
int init_iter = iter;

int frameIdGlobal;

int predictionTimes{0};

/// TODO optimizations:
/// 1. replace sqrt function with inline function of math

int main(int argc, char **argv)
{
    pathTolyftData = *(argv+1);
//    pathTolyftData = "/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/";
    classifierPath = *(argv+2);

    if (init() == 1) {
        std::cout << "Initialization problem" << std::endl;
        return 1;
    }

    LyftDatasetHandler dh_lyft;
    VecArray testData, objectsAboveRoad;

    catToInt["vehicle"] = 0; catToInt["walker"] = 1; catToInt["road"] = 2;
    VecArray vehicle, walker, road;

#if TRAINING == 1
    /// read lyft data
    /*
    std::string JsonFile = hp.getCurrentDir()+jsonLidarSmall;
//    std::string JsonFile = hp.getCurrentDir()+jsonLidarBig;
    std::string readRes = dh_lyft.ReadJson(JsonFile);
    dh_lyft.ParseJson(readRes, pathTolyftData);

    std::vector<Sample> samples = dh_lyft.getSamples();
    std::vector<Annotation> anns = dh_lyft.getAnnotations();

    vec center;
    VecArray debug;
    std::string res = setDataForTraining(pathTolyftData, dh_lyft, anns, center, debug);

    /// choose specific sample
    const int frame_id = 335;
=
    /// show camera images
    Sample current_sample = samples[frame_id];
    showImages("Images", current_sample);

    /// ****************************************************************************************************************
    /// read lidar top data from lyft
    /// ****************************************************************************************************************
    std::string filePathLidarTop = pathTolyftData + samples[frame_id].LidarTopPath;

    std::ifstream infile(filePathLidarTop);
    bool existFile = infile.good();
    if (!existFile) {
        std::cout << "file " << filePathLidarTop << " does not exist\n";
        return 0;
    }

    /// the dataset contains 5 values per point (x, y, z, intensity, ring index)
    int pointsSizeLidarTop = dh_lyft.getBinarySize(filePathLidarTop); /// size is referred to the whole dataset
    auto *pointsLidarTop = new float[pointsSizeLidarTop/sizeof(float)];

    dh_lyft.getBinaryData(filePathLidarTop, pointsSizeLidarTop, pointsLidarTop); /// i need only the 3 of the values of the dataset (x, y, z). So the
    /// points are referred to all of them(x, y, z), leaving the last 2 columns

    /// I need only the first three values so i need to update the size, leaving out the last 2 columns (intensity, ring index)
    pointsSizeLidarTop -= 2*(pointsSizeLidarTop/5);

    /// transform lidar data to global coordinates (into the ego vehicle frame)
    dh_lyft.TransformToGlobalCoord(pointsLidarTop, pointsSizeLidarTop/sizeof(float), vec(egoPos.x, egoPos.y, egoPos.z));
    /// ****************************************************************************************************************
    */

    /// ************************************************************************************************************
    /// read lidar top data from carla
    /// ************************************************************************************************************
    VecArray points, debug;

    std::string out_path = "/home/andreas/Desktop/training_data/";
    std::string pathToMetaData{out_path + "metadata_data.bin"};
    std::string storage_path = out_path + "res/image/";
    vec camera_pos, lidar_pos;
    VecArray vehicle_pos, extrinsics;
    std::vector<bool> vehicle_moving;
    readMetaData(pathToMetaData, camera_pos, lidar_pos, vehicle_pos, extrinsics, vehicle_moving);
    std::string prefix;

    /// exclude first frame
    int init_frame{11806};
    int last_frame{11836};
    int frame_step{30}; /// set only one frame
    std::string pathToData;
    glm::vec3 rotationAxis2(0.0, 0.0, -1.0);
    VecArray vehicleCorners, walkerCorners, roadCorners;
    int iteration{1}, limit{0};

    vehicle.clear();
    walker.clear();
    road.clear();
    std::stringstream trainingData;
    trainingData << "type eigenvaluesSum omnivariance eigenentropy linearity planarity sphericity curvatureChange "
                    "verticalityFirstEigenvectorAxisZ verticalityThirdEigenvectorAxisZ absoluteMomentFirstOrderE1 "
                    "absoluteMomentFirstOrderE2 absoluteMomentFirstOrderE3 absoluteMomentSecondOrderE1 "
                    "absoluteMomentSecondOrderE2 absoluteMomentSecondOrderE3 verticalMomentFirstOrder "
                    "verticalMomentSecondOrder pointsNumber\n";

    int numVehicles{0}, numWalkers{0}, numRoads{0};
//    int thresVehicles{500}, thresWalkers{500}, thresRoads{500};
    int thresVehicles{1}, thresWalkers{1};
    Helpers hp;

    for (unsigned int i=init_frame; i<last_frame; i+=frame_step) {
        std::cout << "frame => " << i << std::endl;

        points.clear();

        if (i <= limit)
            prefix = "/00";
        else
            prefix = "/0";

        pathToData = out_path + "lidar_bin" + prefix + std::to_string(i) + ".bin";
        std::cout << "pathToData = " << pathToData << std::endl;

        readData(pathToData, points);

        float radian = extrinsics[iteration].y*M_PI/180.0; /// convert radian to degree
        hp.rotate_data(points, radian, rotationAxis2);

        vehicleCorners.clear();
        walkerCorners.clear();
        roadCorners.clear();

        std::string pathToVehicles = out_path + "vehicles/" + std::to_string(iteration+1) + ".bin";
        hp.getBoxes(vehicleCorners, pathToVehicles);

        std::string pathToWalkers = out_path + "walkers/" + std::to_string(iteration+1) + ".bin";
        hp.getBoxes(walkerCorners, pathToWalkers);

        std::string pathToRoads = out_path + "roads/" + std::to_string(iteration+1) + ".bin";
        hp.getBoxes(roadCorners, pathToRoads);

        int minPointsVehicle{400}, minPointsWalker{30}, minPointsRoad{100};

        /// find points of vehicles inside box
        VecArray eachBox;

//        for (unsigned int i=0; i<vehicleCorners.size(); i+=8) {
//            eachBox.clear();
//            eachBox.emplace_back(vehicleCorners[i]); eachBox.emplace_back(vehicleCorners[i+1]);
//            eachBox.emplace_back(vehicleCorners[i+2]); eachBox.emplace_back(vehicleCorners[i+3]);
//            eachBox.emplace_back(vehicleCorners[i+4]); eachBox.emplace_back(vehicleCorners[i+5]);
//            eachBox.emplace_back(vehicleCorners[i+6]); eachBox.emplace_back(vehicleCorners[i+7]);
//
//            /// find the points that fall in each box
//            int num{0};
//            VecArray tmpV;
//            for (auto p: points) {
//                if (hp.pointInBox(p, eachBox)) {
//                    tmpV.emplace_back(p);
//                    num++;
//                }
//            }
//            if (num >= minPointsVehicle) {
//                if (numVehicles < thresVehicles) {
//
//                    /// to remove
//                    vehicle.clear();
//
//                    for (auto& v : tmpV)
//                        vehicle.emplace_back(v);
//
//                    /// extract features for this object
//                    debug.clear();
//                    Box3D box(eachBox, "vehicle");
//                    setCarlaDataForTraining(vehicle, box, debug, trainingData);
//                    numVehicles++;
//                }
//            }
//        }


        /// find points of walkers inside box
        for (unsigned int i=0; i<walkerCorners.size(); i+=8) {
            eachBox.clear();
            eachBox.emplace_back(walkerCorners[i]); eachBox.emplace_back(walkerCorners[i+1]);
            eachBox.emplace_back(walkerCorners[i+2]); eachBox.emplace_back(walkerCorners[i+3]);
            eachBox.emplace_back(walkerCorners[i+4]); eachBox.emplace_back(walkerCorners[i+5]);
            eachBox.emplace_back(walkerCorners[i+6]); eachBox.emplace_back(walkerCorners[i+7]);

            /// find the points that fall in each box
            int num{0};
            VecArray tmpW;
            for (auto p: points) {
                if (hp.pointInBox(p, eachBox)) {
                    tmpW.emplace_back(p);
                    num++;
                }
            }
            if (num >= minPointsWalker) {
                if (numWalkers < thresWalkers) {
                    for (auto &v : tmpW)
                        walker.emplace_back(v);

                    /// extract features for this object
                    debug.clear();
                    Box3D box(eachBox, "walker");
                    setCarlaDataForTraining(walker, box, debug, trainingData);
                    numWalkers++;
                }
            }
        }
//
//        /// find points of roads inside box
//        for (unsigned int i=0; i<roadCorners.size(); i+=8) {
//            eachBox.clear();
//            eachBox.emplace_back(roadCorners[i]); eachBox.emplace_back(roadCorners[i+1]);
//            eachBox.emplace_back(roadCorners[i+2]); eachBox.emplace_back(roadCorners[i+3]);
//            eachBox.emplace_back(roadCorners[i+4]); eachBox.emplace_back(roadCorners[i+5]);
//            eachBox.emplace_back(roadCorners[i+6]); eachBox.emplace_back(roadCorners[i+7]);
//
//            /// find the points that fall in each box
//            int num{0};
//            VecArray tmpR;
//            for (auto p: points) {
//                if (hp.pointInBox(p, eachBox)) {
//                    if (p.z <= -2.9) { /// TODO need to validate function hp.pointInBox in order to remove this threshold
//                        tmpR.emplace_back(p);
//                        num++;
//                    }
//                }
//            }
//            if (num >= minPointsRoad) {
//                if (numRoads < thresRoads) {
//                    for (auto& v : tmpR)
//                        road.emplace_back(v);
//
//                    /// extract features for this object
//                    debug.clear();
//                    Box3D box(eachBox, "road");
//                    setCarlaDataForTraining(road, box, debug, trainingData);
//                    numRoads++;
//                }
//            }
//        }
//        iteration++;
    }

    std::cout << "debug size = " << debug.size() << std::endl;

    /// store results to file
//    std::ofstream myfile;
//    myfile.open ("data/trainingData.dat");
//    myfile << trainingData.str();
//    myfile.close();
    /// ************************************************************************************************************
#endif

#if TESTING == 1
    /// read lyft data
    /*
    std::string JsonFile = hp.getCurrentDir()+jsonLidarSmall;
//    std::string JsonFile = hp.getCurrentDir()+jsonLidarBig;
    std::string readRes = dh_lyft.ReadJson(JsonFile);
    dh_lyft.ParseJson(readRes, pathTolyftData);

    std::vector<Sample> samples = dh_lyft.getSamples();
    std::vector<Annotation> anns = dh_lyft.getAnnotations();

    vec center;
    VecArray debug;
    std::string res = setDataForTraining(pathTolyftData, dh_lyft, anns, center, debug);

    /// choose specific sample
    const int frame_id = 335;
//    const int frame_id = 255;

    /// show camera images
    Sample current_sample = samples[frame_id];
    showImages("Images", current_sample);

    /// ****************************************************************************************************************
    /// read lidar top data from lyft
    /// ****************************************************************************************************************
    std::string filePathLidarTop = pathTolyftData + samples[frame_id].LidarTopPath;

    std::ifstream infile(filePathLidarTop);
    bool existFile = infile.good();
    if (!existFile) {
        std::cout << "file " << filePathLidarTop << " does not exist\n";
        return 0;
    }

    /// the dataset contains 5 values per point (x, y, z, intensity, ring index)
    int pointsSizeLidarTop = dh_lyft.getBinarySize(filePathLidarTop); /// size is referred to the whole dataset
    auto *pointsLidarTop = new float[pointsSizeLidarTop/sizeof(float)];

    dh_lyft.getBinaryData(filePathLidarTop, pointsSizeLidarTop, pointsLidarTop); /// i need only the 3 of the values of the dataset (x, y, z). So the
    /// points are referred to all of them(x, y, z), leaving the last 2 columns

    /// I need only the first three values so i need to update the size, leaving out the last 2 columns (intensity, ring index)
    pointsSizeLidarTop -= 2*(pointsSizeLidarTop/5);

    /// transform lidar data to global coordinates (into the ego vehicle frame)
    dh_lyft.TransformToGlobalCoord(pointsLidarTop, pointsSizeLidarTop/sizeof(float), vec(egoPos.x, egoPos.y, egoPos.z));
    /// ****************************************************************************************************************
    */

    /// ************************************************************************************************************
    /// read lidar top data from carla
    /// ************************************************************************************************************
    VecArray points, debug;

    std::string out_path = "/home/andreas/Desktop/training_data/";
    std::string pathToMetaData{out_path + "metadata_data.bin"};
    std::string storage_path = out_path + "res/image/";
    vec camera_pos, lidar_pos;
    VecArray vehicle_pos, extrinsics;
    std::vector<bool> vehicle_moving;
    readMetaData(pathToMetaData, camera_pos, lidar_pos, vehicle_pos, extrinsics, vehicle_moving);
    std::string prefix;

    /// exclude first frame
    int init_frame{11806}, last_frame{61396}, frame_step{30}; /// set only one frame
    std::string pathToData;
    glm::vec3 rotationAxis2(0.0, 0.0, -1.0);
    VecArray vehicleCorners, walkerCorners, roadCorners;
    int iteration{1}, limit{0};

    vehicle.clear();
    walker.clear();
    road.clear();
    std::stringstream trainingData;
    trainingData << "type eigenvaluesSum omnivariance eigenentropy linearity planarity sphericity curvatureChange "
                    "verticalityFirstEigenvectorAxisZ verticalityThirdEigenvectorAxisZ absoluteMomentFirstOrderE1 "
                    "absoluteMomentFirstOrderE2 absoluteMomentFirstOrderE3 absoluteMomentSecondOrderE1 "
                    "absoluteMomentSecondOrderE2 absoluteMomentSecondOrderE3 verticalMomentFirstOrder "
                    "verticalMomentSecondOrder pointsNumber\n";

    int numVehicles{0}, numWalkers{0}, numRoads{0};
//    int thresVehicles{1000}, thresWalkers{1000}, thresRoads{0};
    int thresVehicles{0}, thresWalkers{45}, thresRoads{0};
    Helpers hp;

    int true_vehicle{0}, vehicle_for_walker{0}, vehicle_for_road{0};
    int true_walker{0}, walker_for_vehicle{0}, walker_for_road{0};
    int true_road{0}, road_for_vehicle{0}, road_for_walker{0};

    for (unsigned int i=init_frame; i<last_frame; i+=frame_step) {

        std::cout << "frame => " << i << std::endl;
        std::cout << "vehicles => " << numVehicles << " / " << thresVehicles << std::endl;
        std::cout << "walkers => " << numWalkers << " / " << thresWalkers << std::endl;
        std::cout << "roads => " << numRoads << " / " << thresRoads << std::endl << std::endl;

        if (numVehicles == thresVehicles && numWalkers == thresWalkers && numRoads == thresRoads)
            break;

        points.clear();

        if (i <= limit)
            prefix = "/00";
        else
            prefix = "/0";

        pathToData = out_path + "lidar_bin" + prefix + std::to_string(i) + ".bin";
        readData(pathToData, points);

        float radian = extrinsics[iteration].y*M_PI/180.0; /// convert radian to degree
        hp.rotate_data(points, radian, rotationAxis2);

        vehicleCorners.clear();
        walkerCorners.clear();
        roadCorners.clear();

        std::string pathToVehicles = out_path + "vehicles/" + std::to_string(iteration+1) + ".bin";
        hp.getBoxes(vehicleCorners, pathToVehicles);

        std::string pathToWalkers = out_path + "walkers/" + std::to_string(iteration+1) + ".bin";
        hp.getBoxes(walkerCorners, pathToWalkers);

        std::string pathToRoads = out_path + "roads/" + std::to_string(iteration+1) + ".bin";
        hp.getBoxes(roadCorners, pathToRoads);

        int minPointsVehicle{500}, minPointsWalker{5}, minPointsRoad{1000};

        /// find points of vehicles inside box
        VecArray eachBox;
        std::vector<int> categories;

//        for (unsigned int i=0; i<vehicleCorners.size(); i+=8) {
//            eachBox.clear();
//            eachBox.emplace_back(vehicleCorners[i]); eachBox.emplace_back(vehicleCorners[i+1]);
//            eachBox.emplace_back(vehicleCorners[i+2]); eachBox.emplace_back(vehicleCorners[i+3]);
//            eachBox.emplace_back(vehicleCorners[i+4]); eachBox.emplace_back(vehicleCorners[i+5]);
//            eachBox.emplace_back(vehicleCorners[i+6]); eachBox.emplace_back(vehicleCorners[i+7]);
//
//            /// find the points that fall in each box
//            int num{0};
//            VecArray tmpV;
//            for (auto p: points) {
//                if (hp.pointInBox(p, eachBox)) {
//                    tmpV.emplace_back(p);
//                    num++;
//                }
//            }
////            std::cout << "num = " << num << std::endl;
//            if (num >= minPointsVehicle) {
//                if (numVehicles < thresVehicles) {
//                    vehicle.clear();
//
//                    for (auto& v : tmpV)
//                        vehicle.emplace_back(v);
//
//                    /// extract features for this object
//                    categories.clear();
//                    Classifier cl(classifierPath);
//
//                    int res = cl.run(vehicle);
//                    /// vehicle
//                    if (res == 0)
//                        true_vehicle++;
//                    /// walker
//                    if (res == 1)
//                        vehicle_for_walker++;
//                    /// road
//                    if (res == 2)
//                        vehicle_for_road++;
//
//                    numVehicles++;
//                }
//            }
//        }

        /// find points of walkers inside box
        for (unsigned int i=0; i<walkerCorners.size(); i+=8) {
            eachBox.clear();
            eachBox.emplace_back(walkerCorners[i]); eachBox.emplace_back(walkerCorners[i+1]);
            eachBox.emplace_back(walkerCorners[i+2]); eachBox.emplace_back(walkerCorners[i+3]);
            eachBox.emplace_back(walkerCorners[i+4]); eachBox.emplace_back(walkerCorners[i+5]);
            eachBox.emplace_back(walkerCorners[i+6]); eachBox.emplace_back(walkerCorners[i+7]);

            /// find the points that fall in each box
            int num{0};
            VecArray tmpW;
            for (auto p: points) {
                if (hp.pointInBox(p, eachBox)) {
                    tmpW.emplace_back(p);
                    num++;
                }
            }
            if (num >= minPointsWalker) {
                if (numWalkers < thresWalkers) {
                    walker.clear();

                    for (auto &v : tmpW)
                        walker.emplace_back(v);

                    /// extract features for this object
                    categories.clear();
                    Classifier cl(classifierPath);
                    int res = cl.run(walker);
                    /// vehicle
                    if (res == 0)
                        walker_for_vehicle++;
                    /// walker
                    if (res == 1)
                        true_walker++;
                    /// road
                    if (res == 2)
                        walker_for_road++;

                    numWalkers++;
                }
            }
        }

//        /// find points of roads inside box
//        for (unsigned int i=0; i<roadCorners.size(); i+=8) {
//            eachBox.clear();
//            eachBox.emplace_back(roadCorners[i]); eachBox.emplace_back(roadCorners[i+1]);
//            eachBox.emplace_back(roadCorners[i+2]); eachBox.emplace_back(roadCorners[i+3]);
//            eachBox.emplace_back(roadCorners[i+4]); eachBox.emplace_back(roadCorners[i+5]);
//            eachBox.emplace_back(roadCorners[i+6]); eachBox.emplace_back(roadCorners[i+7]);
//
//            /// find the points that fall in each box
//            int num{0};
//            VecArray tmpR;
//            for (auto p: points) {
//                if (hp.pointInBox(p, eachBox)) {
//                    if (p.z <= -2.9) { /// TODO need to validate function hp.pointInBox in order to remove this threshold
//                        tmpR.emplace_back(p);
//                        num++;
//                    }
//                }
//            }
//            if (num >= minPointsRoad) {
//                if (numRoads < thresRoads) {
//                    road.clear();
//
//                    for (auto& v : tmpR)
//                        road.emplace_back(v);
//
//                    /// extract features for this object
//                    categories.clear();
//                    Classifier cl(classifierPath);
//                    int res = cl.run(road);
//                    /// vehicle
//                    if (res == 0)
//                        road_for_vehicle++;
//                    /// walker
//                    if (res == 1)
//                        road_for_walker++;
//                    /// road
//                    if (res == 2)
//                        true_road++;
//
//                    numRoads++;
//                }
//            }
//        }


        iteration++;
    }


    std::cout << "from " << numVehicles << " num of vechicles : " << " true vehicles = " << true_vehicle << ", " <<
                                                                      " vehicle as walker = " << vehicle_for_walker << ", " <<
                                                                      " vehicle as road = " << vehicle_for_road << std::endl;

    std::cout << "from " << numWalkers << " num of walkers : " << " true walkers = " << true_walker << ", " <<
                                                                      " walker as vehicle = " << walker_for_vehicle << ", " <<
                                                                      " walker as road = " << walker_for_road << std::endl;

    std::cout << "from " << numRoads << " num of roads : " << " true roads = " << true_road << ", " <<
                                                                      " road as vehicle = " << road_for_vehicle << ", " <<
                                                                      " road as walker = " << road_for_walker << std::endl;

    /// ************************************************************************************************************
#endif

#if SIMPLE_RUN == 1
    std::vector<std::string> images;
    VecArray curbPoints, eachCurbPoints;

    /// road segmentation
    VecArray totalGroundV1, totalNoGroundV1, allSegmentsV1;
    VecArray groundV2, noGroundV2, totalGroundV2, totalNoGroundV2;
    std::vector<int> lengthEachSegV1;

    /// rings extraction
    VecArray projections, ringsCols;
    vec projectionPoint = vec(0, 0, 0.05);

    glm::vec3 gEgoPos;
    VecArray points, allPoints, roadPointsVector;

    /// read ego locations
    VecArray ego_locations;
    std::ifstream input( "data/ego_locations.txt"); /// tmp
    for( std::string line; getline( input, line ); ) {
        vec p;
        input >> p.x >> p.y >> p.z;
        ego_locations.emplace_back(vec(0.0f, 0.0f, 0.0f));
    }

    VecArray debug, curbPointsToTrack, left_curbPointsToTrack, right_curbPointsToTrack, curbPointsKalman, allCurbPoints, lastCurbPoints, tmpCurbPoints;

    /// kalman
    State x;
    Control u;
    SystemModel sys;

    /// define filter for estimation
    /// Pure predictor without measurement updates
    Kalman::ExtendedKalmanFilter<State> predictor;
    Clustering clustering;
    VecArray roadLinesUp, roadLinesBelow;
    glm::vec3 record_vec;

    /// store images name and image pixel to evaluate road detection algorithm
    std::vector<std::pair<std::string, int>> pairSI;
    std::string filename;

    /// first frame should be in straight (clear) line in order for the clustering to extract two big clusters and other smaller
    /// so to get the two first biggest clusters as the two sides of the road. So for the next frames, we will use
    /// the centroids of these clusters to choose the new clusters if more than two clusters will be found again..

    /// TODO : check if the quality of first frame is big enough, otherwise wait until another frame comes with good quality

    vec last_vehicle_pos;
    bool stopped, predict;
    VecArray curbUp, curbBelow;
    pairIFF *ringIndices = nullptr;

#if RECORD == 1
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
    while (!glfwWindowShouldClose(window))
    {
#endif

    std::string out_path;
    std::string storage_path;
    VecArray vehicle_pos, extrinsics;
    std::vector<bool> vehicle_moving;
    int limit, step;
    std::string prefix;

    vehicle_pos.clear();
    extrinsics.clear();
    vehicle_moving.clear();

    int rings;
    #if CARLA_DATA == 1
        record_vec = glm::vec3(12.0f, 0.0f, 25.0f);
        unsigned int occl=5;
        out_path = "/media/andreas/Storage/Downloads/Town01_07_10_2020_11_52_12/";
        frameStart = 3728;
        frameEnd = 3818;
        iter = 1035;
        limit = 9998;
        step = 30;
        storage_path = out_path + "res/" + std::to_string(occl);
        rings = 60; /// old
    #endif

        Helpers hp;

    #if LYFT_DATA == 1
        record_vec = glm::vec3(15.0f, 0.0f, 35.0f);
        unsigned int occl=5;
        out_path = "/home/andreas/Desktop/test/";

        frameStart = 0;
        frameEnd = 5074;

        iter = 1;
        limit = 9998;
        step = 1;
        storage_path = out_path + "res/image/";
        rings = 15;

        std::string JsonFile = hp.getCurrentDir()+jsonLidarSmall;
//        std::string JsonFile = hp.getCurrentDir()+jsonLidarMedium;
//        std::string JsonFile = hp.getCurrentDir()+jsonLidarBig;

        std::string readRes = dh_lyft.ReadJson(JsonFile);
        dh_lyft.ParseJson(readRes, pathTolyftData);

        std::vector<Sample> samples = dh_lyft.getSamples();
        std::vector<Annotation> anns = dh_lyft.getAnnotations();
    #endif

    #if CARLA_DATA == 1
        std::string pathToMetaData{out_path + "metadata_data.bin"};
//        storage_path = out_path + "res/image/";
        vec camera_pos, lidar_pos;
        readMetaData(pathToMetaData, camera_pos, lidar_pos, vehicle_pos, extrinsics, vehicle_moving);
    #endif


    init_iter = iter;

    unsigned int frame_id=frameStart;

    std::vector<std::string> all_paths;
//    std::string path = "/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/lidar";
//    for (const auto & entry : fs::directory_iterator(path))
//        all_paths.emplace_back(entry.path());

//    std::ifstream infile("/home/andreas/Desktop/all_lidar_paths.txt");
//    std::string line;
//    while (std::getline(infile, line)) {
//        all_paths.emplace_back(line);
//    }

//    for (unsigned int frame_id=0; frame_id<all_paths.size(); frame_id+=step) {
    for (unsigned int frame_id=frameStart; frame_id<frameEnd; frame_id+=step) {
        CurbDetection curbDetect;
        stopped = false;

        frameIdGlobal = frame_id;

        std::cout << "\nframe_id = " << frame_id << std::endl;

        # if CARLA_DATA == 1
            if (!vehicle_moving[iter]) {
                std::cout << "vehicle has stopped!!\n\n";
                stopped = true;
            }
        #endif

        /// read lyft data
        #if LYFT_DATA == 1
            /// ************************************************************************************************************
            /// read lidar top data from lyft
            /// ************************************************************************************************************
//            std::string filePathLidarTop = pathTolyftData + samples[frame_id].LidarTopPath;
//            std::cout << "yaw = " << samples[frame_id].yaw << std::endl;

//            std::string filePathLidarTop = "/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/lidar/host-a101_lidar1_1241216089202636746.bin";
//            std::string filePathLidarTop = "/home/andreas/Desktop/data/input_lidar_points.bin";
            std::string filePathLidarTop = all_paths[frame_id];

            std::cout << "yaw = " << samples[frame_id].yaw << std::endl;
            std::cout << "pitch = " << samples[frame_id].pitch << std::endl;
            std::cout << "roll = " << samples[frame_id].roll << std::endl;
            std::cout << "lidar_top_rotation = " << std::endl;
            points.clear();
            for (auto& val : samples[frame_id].lidar_top_rotation) {
                std::cout << val.x << std::endl;
                std::cout << val.y << std::endl;
                std::cout << val.z << std::endl;

                points.emplace_back(vec(val.x, val.y, val.z));
            }

            std::ifstream infile(filePathLidarTop);
            bool existFile = infile.good();
            if (!existFile) {
                std::cout << "file " << filePathLidarTop << " does not exist\n";
                #if LYFT_DATA == 1
                std::string filename_p = "/home/andreas/Desktop/data/output_lidar_points_"+ std::to_string(frame_id) +".bin";
                std::string filename_c = "/home/andreas/Desktop/data/output_lidar_colours_"+ std::to_string(frame_id) +".bin";
                VecArray final_points, final_colours;
                write_to_bin(final_points, filename_p, 0);
                write_to_bin(final_colours, filename_c, 1);
                #endif

                continue;
            }

            /// the dataset contains 5 values per point (x, y, z, intensity, ring index)
            int pointsSizeLidarTop = dh_lyft.getBinarySize(filePathLidarTop); /// size is referred to the whole dataset
            auto *pointsLidarTop = new float[pointsSizeLidarTop/sizeof(float)];

            points.clear();
            dh_lyft.getBinaryData(filePathLidarTop, pointsSizeLidarTop, pointsLidarTop, points); /// i need only the 3 of the values of the dataset (x, y, z). So the
            /// points are referred to all of them(x, y, z), leaving the last 2 columns

            glm::vec3 rotationAxis(0.0, 0.0, -1.0);
            hp.rotate_data(points, 3.141592f, rotationAxis); /// rotate point cloud to calibrate it with the camera sensor
            /// ************************************************************************************************************
        #endif

        #if CARLA_DATA == 1
            /// ************************************************************************************************************
            /// read lidar top data from carla
            /// ************************************************************************************************************
            images.clear();
            std::string pathIm = out_path + "image";

            if (frame_id <= limit)
                prefix = "/00";
            else
                prefix = "/0";

            images.emplace_back(pathIm + prefix + std::to_string(frame_id) + ".png");

            std::string pathToData;
            std::string pathLidar = out_path + "lidar_bin";
            pathToData = pathLidar + prefix + std::to_string(frame_id) + ".bin";

            points.clear();
            readData(pathToData, points);
        #endif

//        glm::vec3 rotationAxis(0.0, 0.0, -1.0);
//        hp.rotate_data(points, 1.57f, rotationAxis); /// rotate point cloud to calibrate it with the camera sensor

        /// ************************************************************************************************************
        /// road segmentation on original data
        /// ************************************************************************************************************
        std::cout << "road segmentation...\n";
        unsigned int Nseg;
        Segmentation segmentation;
        segmentation.setCloud(points);

        #if CARLA_DATA == 1
            Nseg = 10;
            segmentation.init(10, 50, 0.01, 0.25f, Nseg, points);
        #endif

        #if LYFT_DATA == 1
            Nseg = 30;
            segmentation.init(50, 30, 0.2, 0.5f, Nseg, points);
        #endif

        totalGroundV1.clear(); totalNoGroundV1.clear(); allSegmentsV1.clear(); lengthEachSegV1.clear();
        segmentation.segmentCloud(totalGroundV1, totalNoGroundV1, allSegmentsV1, lengthEachSegV1);
        /// ************************************************************************************************************

        #if LYFT_DATA == 1
            stopped = false;
        #endif

        /// if car is moving
        if (!stopped) {
            /// ************************************************************************************************************
            /// rings extraction
            /// ************************************************************************************************************
            std::cout << "rings extraction...\n";
            curbDetect.initialize(totalGroundV1, rings, temporalEgoPos);
            projections.clear();
            curbDetect.projectionToXY(projections, projectionPoint);
            curbDetect.setProjectionPoints(projections);

            float initialRadius{5.0f}, radiusIncrement{0.5f};
            std::unordered_map<int, int> idToRing;
            bool occlusion;
            curbDetect.m_left_occlusion = false;
            curbDetect.m_right_occlusion = false;
            debug.clear();

            ringIndices = curbDetect.extractRings(initialRadius, radiusIncrement, idToRing, occlusion, debug, occl);

            /// generate ring colors
            int bins = curbDetect.getBins();
            curbDetect.generateColors(bins, ringsCols);
            /// ************************************************************************************************************

            /// ************************************************************************************************************
            /// curb detection
            /// ************************************************************************************************************
            std::cout << "curb detection...\n";
            std::vector<int> curbIds;
            pairIIFV results; /// store region_ring_angle_curb

            eachCurbPoints.clear();
            testData.clear();

            curbDetect.detection(ringIndices, eachCurbPoints, debug, results);

            /// gather all curb points - iterate through all regions and create hash table with each curb point and its ring
            curbPoints.clear();
            VecArray curb_left, curb_right;
            float angle;
            vec p;

            /// store curb points and split them to left and right points
            for (auto &region_ring_angle_curb : results) {
                p = std::get<3>(region_ring_angle_curb);
                curbPoints.emplace_back(p);
                angle = std::get<2>(region_ring_angle_curb);
                if (angle <= 90.0f)
                    curb_left.emplace_back(p);
                else
                    curb_right.emplace_back(p);
            }

            predict = curbDetect.m_occlusion;
            /// ************************************************************************************************************

            /// initialize prediction algorithm in case of occlusion (only for the first frame)
            if (iter == init_iter) {
                left_curbPointsToTrack.clear();
                left_curbPointsToTrack = curb_left;
                right_curbPointsToTrack.clear();
                right_curbPointsToTrack = curb_right;
            }

            #if LYFT_DATA == 1
                predict = false;
            #endif

            /// if there is no occlusion run object detection using Conditional Euclidean Clustering
            if (!predict) {
                debug.clear();
                clustering.curbs_clustering(curbPoints, debug);

                /// store curb points in order to use them for the prediction step if there is occlusion in the next frame
                /// split to curb points to left and right depending on their angles
                left_curbPointsToTrack.clear();
                right_curbPointsToTrack.clear();
                SphericalCoordinates scPoint;
                for (auto&p : curbPoints) {
                    scPoint = curbDetect.convertToSpherical(p);
                    if (scPoint.th <= 90.0f)
                        left_curbPointsToTrack.emplace_back(p);
                    else
                        right_curbPointsToTrack.emplace_back(p);
                }

                if (clustering.m_clusters == 1) {
                    iter+=1;
                    std::cout << "could not find enough curb points";

                    #if LYFT_DATA == 1
                    std::string filename_p = "/home/andreas/Desktop/data/output_lidar_points_"+ std::to_string(frame_id) +".bin";
                    std::string filename_c = "/home/andreas/Desktop/data/output_lidar_colours_"+ std::to_string(frame_id) +".bin";
                    VecArray final_points, final_colours;
                    write_to_bin(final_points, filename_p, 0);
                    write_to_bin(final_colours, filename_c, 1);
                    #endif
                    continue;
                }
            }
            else {
                std::cout << "curb prediction...\n";

                predictionTimes++;

                /// add curb points to track from previous frame
                curbPointsToTrack.clear();

                if (curbDetect.m_left_occlusion)
                    curbPointsToTrack.insert(curbPointsToTrack.end(), left_curbPointsToTrack.begin(), left_curbPointsToTrack.end());
                if (curbDetect.m_right_occlusion)
                    curbPointsToTrack.insert(curbPointsToTrack.end(), right_curbPointsToTrack.begin(), right_curbPointsToTrack.end());

                for (auto &l : curbPointsToTrack) {
                    x.x() = l.x;
                    x.y() = l.y;

                    predictor.init(x);

                    u.dx() = vehicle_pos[iter].x - vehicle_pos[iter-1].x;
                    u.dy() = vehicle_pos[iter].y - vehicle_pos[iter-1].y;
                    u.yaw() = (extrinsics[iter].y - extrinsics[iter-1].y) * M_PI / 180.0f;

//                    u.dx() = 0.0f;
//                    u.dy() = 0.0f;
//                    u.yaw() = 0.0f;

                    /// Simulate system
                    x = sys.f(x, u);

                    /// Predict state for current time-step using the filters
                    auto x_pred = predictor.predict(sys, u);

                    /// store predicted points
                    curbPoints.emplace_back(vec(x_pred.x(), x_pred.y(), -3.0f));
                }

                debug.clear();
                clustering.curbs_clustering(curbPoints, debug);

                /// store curb points in order to use them for the prediction step if there is occlusion in the next frame
                /// split to curb points to left and right depending on their angles
                left_curbPointsToTrack.clear();
                right_curbPointsToTrack.clear();
                SphericalCoordinates scPoint;
                for (auto&p : curbPoints) {
                    scPoint = curbDetect.convertToSpherical(p);
                    if (scPoint.th <= 90.0f)
                        left_curbPointsToTrack.emplace_back(p);
                    else
                        right_curbPointsToTrack.emplace_back(p);
                }

                debug.clear();
                for (auto &p : left_curbPointsToTrack){
                    debug.emplace_back(p);
                    debug.emplace_back(vec(1.0f, 0.0f, 0.0f));
                }
                for (auto &p : right_curbPointsToTrack){
                    debug.emplace_back(p);
                    debug.emplace_back(vec(0.0f, 1.0f, 0.0f));
                }

                if (clustering.m_clusters == 1) {
                    iter+=1;
                    std::cout << "could not find enough curb points";
                    #if LYFT_DATA == 1
                    std::string filename_p = "/home/andreas/Desktop/data/output_lidar_points_"+ std::to_string(frame_id) +".bin";
                    std::string filename_c = "/home/andreas/Desktop/data/output_lidar_colours_"+ std::to_string(frame_id) +".bin";
                    VecArray final_points, final_colours;
                    write_to_bin(final_points, filename_p, 0);
                    write_to_bin(final_colours, filename_c, 1);
                    #endif

                    continue;
                }
            } /// if there is occlusion then predict curb points using the previous ones

            /// store curb points per side
            curbUp.clear();
            curbBelow.clear();
            curbUp = clustering.curbUp;
            curbBelow = clustering.curbBelow;

        } /// if car is moving

        /// estimate road lines
        Eigen::VectorXd roadUp, roadBelow;
        vec tmp_ego_loc = vec(0.0f, 0.0f, 0.0f);
        roadPointsVector.clear();

        debug.clear();
        roadLinesUp.clear();
        roadLinesBelow.clear();

        int draw_road_line = 0;
        if (frame_id == 3398)
            draw_road_line = 1;

        estimateRoadLines(roadUp, roadBelow, curbUp, curbBelow, curbPoints, roadPointsVector, totalGroundV1, totalNoGroundV1, tmp_ego_loc, roadLinesUp, roadLinesBelow, draw_road_line);

        if (roadPointsVector.empty()) {
            iter+=1;
            std::cout << "roadPointsVector is empty";
            #if LYFT_DATA == 1
            std::string filename_p = "/home/andreas/Desktop/data/output_lidar_points_"+ std::to_string(frame_id) +".bin";
            std::string filename_c = "/home/andreas/Desktop/data/output_lidar_colours_"+ std::to_string(frame_id) +".bin";
            VecArray final_points, final_colours;
            write_to_bin(final_points, filename_p, 0);
            write_to_bin(final_colours, filename_c, 1);
            #endif

            continue;
        }

        debug.clear();
        for (unsigned int p=0; p<roadLinesUp.size(); p+=2) {
            debug.emplace_back(roadLinesUp[p]);
            debug.emplace_back(roadLinesUp[p+1]);
        }
        for (unsigned int p=0; p<roadLinesBelow.size(); p+=2) {
            debug.emplace_back(roadLinesBelow[p]);
            debug.emplace_back(roadLinesBelow[p+1]);
        }

        /// ****************************************************************************************************************
        /// road segmentation on data that belong to road only
        /// ****************************************************************************************************************
        Segmentation segmentation2;
        segmentation2.setCloud(roadPointsVector);

        totalGroundV2.clear();
        totalNoGroundV2.clear();

        /// set thresholds in order not to lose curbs
        int Nseg_v2 = 2;

        #if CARLA_DATA == 1
            segmentation2.init(10, 10, 0.01, 0.1f, Nseg_v2, roadPointsVector);
            segmentation2.init(10, 10, 0.01, 0.1f, Nseg_v2, roadPointsVector);
        #endif

        #if LYFT_DATA == 1
            segmentation2.init(50, 10, 0.2, 0.3f, Nseg_v2, roadPointsVector);
        #endif

        VecArray allSegmentsV2;
        std::vector<int> lengthEachSegV2;
        segmentation2.segmentCloud(totalGroundV2, totalNoGroundV2, allSegmentsV2, lengthEachSegV2);
        /// ****************************************************************************************************************

        /// ****************************************************************************************************************
        /// project road data to image
        /// ****************************************************************************************************************
        VecArray imageRoad, imageObjects, imageCurb;
        calibMat calib;
        Helpers hp2;
        #if CARLA_DATA == 1
            calib.fovX = 100.0;
            calib.fovY = 50.0;
            calib.camCenterX = 1248.0;
            calib.camCenterY = 384.0;

            vec v(0.0f, 0.0f, 0.0f);
            hp2.projectToIm(totalGroundV2, imageRoad, calib, v);
            hp2.projectToIm(totalNoGroundV2, imageObjects, calib, v);

            cv::Mat img = cv::imread(images[0]);
        #endif

        #if LYFT_DATA == 1
            vec v(0.0f, 0.0f, 0.0f);
            hp2.projectToIm(totalGroundV2, imageRoad, calib, v, samples[frame_id].ego_pose_rotation[0].x, samples[frame_id].ego_pose_rotation[0].y, samples[frame_id].ego_pose_rotation[0].z);

            filename = samples[frame_id].PathToCameraImages[0];
            std::cout << "filename image = " << filename << std::endl;

            /// bach command to rename jpeg to jpg from folder
            //        for f in *.jpeg; do  mv -- "$f" "${f%.jpeg}.jpg"; done;
            std::cout << "path to image before = " << filename << std::endl;
            filename.replace(filename.begin()+139,filename.end(),"jpg");
            std::cout << "path to image after = " << filename << std::endl;

            cv::Mat img = cv::imread(filename);
        #endif

        int z,y;
        int y_min{10000};

        for (auto &p : imageRoad) {
            z = round(p.x);
            y = round(p.y);
            if (y < y_min)
                y_min = y;
            cv::circle(img, cv::Point(z,y), 0.0, cv::Scalar(128, 64, 128), 17);
        }
        /// ****************************************************************************************************************
        VecArray final_points, final_colours;
        #if LYFT_DATA == 1
            for (auto&p : totalGroundV2) {
                final_points.emplace_back(p);
                final_colours.emplace_back(vec(128.0f/255.0f, 64.0f/255.0f, 128.0f/255.0f));
            }
        #endif

        /// ****************************************************************************************************************
        /// object detection using Conditional Euclidean Clustering
        /// ****************************************************************************************************************
        pcl::PointCloud<PointTypeIO>::Ptr cloud_out (new pcl::PointCloud<PointTypeIO>);
        pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters);
        clustering.objects_clustering(totalNoGroundV2, objectsAboveRoad, calib, v, img, cloud_out, clusters);

        /// classify eact cluster center
        std::vector<int> categories;
        Classifier cl(classifierPath);
        cl.run(cloud_out, clusters, categories);

//        debug.clear();
        objectsAboveRoad.clear();
        int categoryIndex{0}, currentCategory;
        vec eachCol;
        for (int clus = 0; clus < clusters->size (); ++clus) {
            currentCategory = categories[categoryIndex];
            categoryIndex++;

            VecArray object;
            for (int index = 0; index < (*clusters)[clus].indices.size (); ++index) {
                object.emplace_back(vec(cloud_out->points[(*clusters)[clus].indices[index]].x,
                                        cloud_out->points[(*clusters)[clus].indices[index]].y,
                                        cloud_out->points[(*clusters)[clus].indices[index]].z));

                objectsAboveRoad.emplace_back(vec(cloud_out->points[(*clusters)[clus].indices[index]].x,
                                                  cloud_out->points[(*clusters)[clus].indices[index]].y,
                                                  cloud_out->points[(*clusters)[clus].indices[index]].z));
                if (currentCategory == 0)
                    objectsAboveRoad.emplace_back(vec(0.0f, 0.0f, 1.0f));
                if (currentCategory == 1)
                    objectsAboveRoad.emplace_back(vec(1.0f, 0.0f, 0.0f));
            }

            imageObjects.clear();
            #if CARLA_DATA == 1
                hp2.projectToIm(object, imageObjects, calib, v);
            #endif

            #if LYFT_DATA == 1
                hp2.projectToIm(object, imageObjects, calib, v, samples[frame_id].ego_pose_rotation[0].x, samples[frame_id].ego_pose_rotation[0].y, samples[frame_id].ego_pose_rotation[0].z);
            #endif

            for (auto &p : imageObjects) {
                z = round(p.x);
                y = round(p.y);
                if (currentCategory == 0)
                    cv::circle(img, cv::Point(z,y), 1.0, cv::Scalar(255.0f, 0.0f, 0.0f), 10);
                if (currentCategory == 1)
                    cv::circle(img, cv::Point(z,y), 1.0, cv::Scalar(60.0f, 20.0f, 220.0f), 10);
                if (currentCategory == 2)
                    cv::circle(img, cv::Point(z,y), 1.0, cv::Scalar(128, 64, 128), 10);
                if (currentCategory == -1)
                    cv::circle(img, cv::Point(z,y), 1.0, cv::Scalar(0, 0, 0), 10);
            }

            #if LYFT_DATA == 1
                std::string filename_points = "/home/andreas/Desktop/data/output_lidar_points_"+ std::to_string(frame_id) +".bin";
                std::string filename_colours = "/home/andreas/Desktop/data/output_lidar_colours_"+ std::to_string(frame_id) +".bin";
                vec colour;
                if (currentCategory == 0)
                    colour = vec(255.0f/255.0f, 0.0f/255.0f, 0.0f/255.0f);
                if (currentCategory == 1)
                    colour = vec(60.0f/255.0f, 20.0f/255.0f, 220.0f/255.0f);
                for (auto&p : totalNoGroundV2) {
                    final_points.emplace_back(p);
                    final_colours.emplace_back(colour);
                }
                write_to_bin(final_points, filename_points, 0);
                write_to_bin(final_colours, filename_colours, 1);
            #endif
        }


        /// ****************************************************************************************************************

        #if CARLA_DATA == 1
//            std::cout << "store image to " << storage_path + prefix + std::to_string(frame_id)+".png" << std::endl;
//            cv::imwrite(storage_path + prefix + std::to_string(frame_id)+".png", img);
        #endif

        #if LYFT_DATA == 1
//                filename.replace(filename.begin(),filename.begin()+104,"");
//                filename.replace(filename.begin()+35, filename.end(),"png");
//                std::cout << "store image to " << storage_path + filename << std::endl;
//                cv::imwrite(storage_path + filename, img); /// lyft
        #endif

        iter+=1; /// increase counter by two

    #if RECORD == 0
        } /// frames iteration
    #endif

#endif

#if SIMPLE_RUN_WITH_CLASSIFIER_ONLY == 1
    std::vector<std::string> images;
    VecArray curbPoints, eachCurbPoints;

    /// road segmentation
    VecArray totalGroundV1, totalNoGroundV1, allSegmentsV1;
    VecArray groundV2, noGroundV2, totalGroundV2, totalNoGroundV2;
    std::vector<int> lengthEachSegV1;

    /// rings extraction
    VecArray projections, ringsCols;
    vec projectionPoint = vec(0, 0, 0.05); /// carla position ??? to check it

    glm::vec3 gEgoPos;
    VecArray points, allPoints, roadPointsVector;

    /// read ego locations
    VecArray ego_locations;
    std::ifstream input( "data/ego_locations.txt"); /// tmp
    for( std::string line; getline( input, line ); ) {
        vec p;
        input >> p.x >> p.y >> p.z;
        ego_locations.emplace_back(vec(0.0f, 0.0f, 0.0f));
    }

    VecArray debug, curbPointsToTrack, left_curbPointsToTrack, right_curbPointsToTrack, curbPointsKalman, allCurbPoints, lastCurbPoints, tmpCurbPoints;

    /// kalman
    State x;
    Control u;
    SystemModel sys;

    /// define filter for estimation
    /// Pure predictor without measurement updates
    Kalman::ExtendedKalmanFilter<State> predictor;
    Clustering clustering;
    VecArray roadLinesUp, roadLinesBelow;
    glm::vec3 record_vec;

    /// store images name and image pixel to evaluate road detection algorithm
    std::vector<std::pair<std::string, int>> pairSI;

    /// first frame should be in straight (clear) line in order for the clustering to extract two big clusters and other smaller
    /// so to get the two first biggest clusters as the two sides of the road. So for the next frames, we will use
    /// the centroids of these clusters to choose the new clusters if more than two clusters will be found again..

    /// TODO : check if the quality of first frame is big enough, otherwise wait until another frame comes with good quality

    vec last_vehicle_pos;
    bool stopped, predict;
    VecArray curbUp, curbBelow;
    pairIFF *ringIndices = nullptr;

#if RECORD == 1
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
    while (!glfwWindowShouldClose(window))
    {
#endif


    std::string out_path;
    std::string storage_path;
    VecArray vehicle_pos, extrinsics;
    std::vector<bool> vehicle_moving;
    int limit, step;
    std::string prefix;

    for (int town=0; town<1; town++) {
        vehicle_pos.clear();
        extrinsics.clear();
        vehicle_moving.clear();

        record_vec = glm::vec3(12.0f, 0.0f, 25.0f);
        out_path = "/media/andreas/lab/all_thesis_data/Town01_07_10_2020_11_52_12/";
        frameStart = 2828;
        frameEnd = 2858;
        iter = 3;
        limit = 9998;

        std::string pathToMetaData{out_path + "metadata_data.bin"};
        storage_path = out_path + "res/image/";
        vec camera_pos, lidar_pos;
        readMetaData(pathToMetaData, camera_pos, lidar_pos, vehicle_pos, extrinsics, vehicle_moving);

        unsigned int occl=5;
        storage_path = out_path + "res/" + std::to_string(occl);

        const int rings{60};

        init_iter = iter;
        step = 30;

        for (unsigned int frame_id=frameStart; frame_id<frameEnd; frame_id+=step) {
            CurbDetection curbDetect;
            stopped = false;

            frameIdGlobal = frame_id;

            std::cout << "\nframe_id = " << frame_id << std::endl;

            if (!vehicle_moving[iter]) {
                std::cout << "vehicle has stopped!!\n\n";
                stopped = true;
            }
            /// ************************************************************************************************************
            /// read lidar top data from carla
            /// ************************************************************************************************************
            images.clear();
            std::string pathIm = out_path + "image";

            if (town == 6) {
                if (frame_id <= limit)
                    prefix = "/0";
                else
                    prefix = "/";
            }
            else {
                if (frame_id <= limit)
                    prefix = "/00";
                else
                    prefix = "/0";
            }

            images.emplace_back(pathIm + prefix + std::to_string(frame_id) + ".png");

            std::string pathToData;
            std::string pathLidar = out_path + "lidar_bin";
            pathToData = pathLidar + prefix + std::to_string(frame_id) + ".bin";

            points.clear();
            readData(pathToData, points);
            /// ************************************************************************************************************

            /// ************************************************************************************************************
            /// road segmentation on original data
            /// ************************************************************************************************************
            std::cout << "road segmentation...\n";
            unsigned int Nseg{10};
            Segmentation segmentation;
            segmentation.setCloud(points);
            segmentation.init(10, 50, 0.01, 0.25f, Nseg, points); /// old
            totalGroundV1.clear(); totalNoGroundV1.clear(); allSegmentsV1.clear(); lengthEachSegV1.clear();
            segmentation.segmentCloud(totalGroundV1, totalNoGroundV1, allSegmentsV1, lengthEachSegV1);
            /// ************************************************************************************************************


            std::cout << "classification...\n";

            Classifier cl(classifierPath);
            std::vector<int> categories;
            cl.run(totalNoGroundV1, categories);

//            catToInt["vehicle"] = 0; catToInt["walker"] = 1; catToInt["road"] = 2;

            debug.clear();
            int index{0};
            for (auto &p: points) {
                if (categories[index] == 0) {
                    debug.emplace_back(p);
                    debug.emplace_back(vec(0.0f, 0.0f, 1.0f));
                }
                if (categories[index] == 1) {
                    debug.emplace_back(p);
                    debug.emplace_back(vec(1.0f, 0.0f, 0.0f));
                }
                if (categories[index] == 2) {
                    debug.emplace_back(p);
                    debug.emplace_back(vec(0.0f, 1.0f, 0.0f));
                }
            }


//            imageObjects.clear();
//            hp2.projectToIm(object, imageObjects, calib, v);
//            for (auto &p : imageObjects) {
//                z = round(p.x);
//                y = round(p.y);
//                cv::circle(img, cv::Point(z,y), 1.0, cv::Scalar(128, 64, 128), 10);
//            }
            /// ****************************************************************************************************************

//            cv::imwrite(storage_path + prefix + std::to_string(frame_id)+".png", img);

            iter+=1; /// increase counter by two

#if RECORD == 0
        } /// frames iteration
#endif

//        } /// occlusion parameter



        /// TODO add specific value when no object (road/car/walker) has been found
        std::ofstream file_res;
        file_res.open ("data/file_res.txt");
        std::string data_to_write;
        for(auto &p : pairSI) {
            data_to_write = p.first + " " + std::to_string(p.second) + "\n";
            file_res << data_to_write;
        }
        file_res.close();
#endif

    /// build and compile our shader program
    Shader shader("res/shaders/Basic.shader");
    Shader shaderOriginal("res/shaders/Original.shader");
    Shader shaderBox("res/shaders/Box.shader");
    Shader shaderNeighbors("res/shaders/Neighbors.shader");
    Shader shaderRoad("res/shaders/Road.shader");
    Shader shaderPlane("res/shaders/Plane.shader");
    Shader shaderBoundaries("res/shaders/Boundaries.shader");
    Shader shaderObject("res/shaders/Object.shader");
    Shader shaderPolygon("res/shaders/Polygon.shader");
    Shader shaderCurb("res/shaders/Curb.shader");
    Shader shaderProjection("res/shaders/Projection.shader");
    Shader shaderTest("res/shaders/Test.shader");
    Shader shaderRoadPoints("res/shaders/RoadPoints.shader");
    Shader shaderRoadLine("res/shaders/RoadLine.shader");
    Shader shaderGround("res/shaders/Ground.shader");
    Shader shaderSegments("res/shaders/Segments.shader");
    Shader shaderOutliers("res/shaders/Outliers.shader");
    Shader shaderPointsInBoxes("res/shaders/PointInBoxes.shader");
    Shader shaderSpherical("res/shaders/Spherical.shader");
    Shader shaderNonGround("res/shaders/nonGround.shader");
    Shader shaderOrien("res/shaders/Orien.shader");
    Shader shaderVehicles("res/shaders/Vehicle.shader");
    Shader shaderWalkers("res/shaders/Walker.shader");
    Shader shaderRoadCarla("res/shaders/Road_carla.shader");
    Shader shaderObjects("res/shaders/ObjectsAboveRoad.shader");

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glPointSize(2.0);
//    glPointSize(5.5);

#if DRAW_SEGMENTS_V1 == 1
    VecArray segCols;
    CurbDetection curbDetect_2;
    curbDetect_2.generateColors(lengthEachSegV1.size(), segCols);
#endif
#if DRAW_SEGMENTS_V2 == 1
    VecArray segCols;
        curbDetect.generateColors(lengthEachSegV2.size(), segCols);
#endif

#if RECORD == 0
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
    while (!glfwWindowShouldClose(window))
    {
#endif
        /// ***************************************************************
        /// handling window
        /// ***************************************************************
        /// per-frame time logic
        float currentFrame = glfwGetTime();
        camera.deltaTime = currentFrame - camera.lastFrame;
        camera.lastFrame = currentFrame;

        /// input
        camera.processInput(window);

        /// render
        renderer.Clear();

        shader.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        projection = glm::perspective(glm::radians(camera.m_Fov), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 300.0f);
        shader.SetMat4("projection", projection);
        /// camera/view transformation
        view = camera.GetViewMatrix();
        shader.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        model = glm::mat4(1.0f); /// make sure to initialize matrix to identity matrix first
        shader.SetMat4("model", model);

#if RECORD == 1
        camera.setEgoPos(record_vec);
        frameIdGlobal -= step;
        std::string str = out_path + "res/lidar" + prefix + std::to_string(frameIdGlobal) + ".png";
//        std::string str = out_path + "res/lidar/" + filename;

//        std::string str = "/home/andreas/Desktop/data/lidar/" + std::to_string(frameIdGlobal);
        std::cout << "store to " << str << std::endl;

        const char *c = str.c_str();
        saveImage(c, window);
#endif

#if SPHERICAL_NEIGHBORHOOD == 1
        /*
    std::vector<types> categories;
                std::stringstream trainingData;

                /// define point cloud
                pointCloudBoost cloud(new pointCloud), neighbors(new pointCloud);

                /// resize space for point cloud
                cloud->width = (pointsSize/dimensions)/sizeof(float); cloud->height = 1;
                cloud->points.resize (cloud->width * cloud->height);

                /// fill point cloud with data
                int indPoint{0};
                for (auto & point : cloud->points) {
                    point.x = *(points+indPoint); point.y = *(points+indPoint+1); point.z = *(points+indPoint+2);
                    indPoint+=3;
                }

                SphericalNeighborhood sNeighbors;

                /// TODO must optimize these parameters => (initialR, smallestRadius, ratio, scalesNum)
                float initialR{15.0f}, smallestRadius{0.6f}, ratio{1.25f};
                unsigned int scalesNum{12};
                sNeighbors.init(initialR, scalesNum, smallestRadius, ratio, cloud);

                std::vector<GeometricFeatures> geometricFeatures;
                VecArray boxCorners;
                int index;
                vec c;

                trainingData << "type eigenvaluesSum omnivariance eigenentropy linearity planarity sphericity curvatureChange "
                                "verticalityFirstEigenvectorAxisZ verticalityThirdEigenvectorAxisZ absoluteMomentFirstOrderE1 "
                                "absoluteMomentFirstOrderE2 absoluteMomentFirstOrderE3 absoluteMomentSecondOrderE1 "
                                "absoluteMomentSecondOrderE2 absoluteMomentSecondOrderE3 verticalMomentFirstOrder "
                                "verticalMomentSecondOrder pointsNumber\n";

                for (unsigned int i=0; i<boxes.size(); i++) {
                    //           std::cout << "new box and cloud size = " << cloud->points.size() << std::endl;
                    boxCorners = boxes[i].corners;
                    c = getCenter(boxCorners);
                    sNeighbors.setCenter(c);
                    sNeighbors.setCloud(cloud);

                    if (existInVector(categories, boxes[i].category) == -1) {
                        categories.emplace_back(boxes[i].category, categories.size());
                        index = categories.size()-1;
                    }
                    else {
                        for (auto &p : categories)
                            if (p.name == boxes[i].category)
                                index = p.index;
                    }

                    sNeighbors.run(geometricFeatures, trainingData, index);
                }

                std::ofstream myfile;
                myfile.open ("trainingData.dat");
                myfile << trainingData.str();
                myfile.close();

                float *neighborsPoints;
                int neighborsSize{0};
                #if DRAW_NEIGHBORS == 1
                    sNeighbors.getNeighbors(neighbors);
                            int NumNeighbors = neighbors->width*neighbors->height;
                            int rgbValues{3};
                            neighborsSize = (dimensions + rgbValues) * NumNeighbors * sizeof(float); /// dimensions declare that we have 3d points

                            /// increase neighborsSize in order to add the center point with its color
                            neighborsSize += 6* sizeof(float);
                            /// neighborsPoints stores all neighbors (3d points with its rgb color) and leaves more 6 floats empty for the center point

                            neighborsPoints = renderer.setData(neighbors, vec(1.0f, 0.0f, 0.0f), neighborsSize/sizeof(float));

                            /// add center to neighborsPoints and draw using another color
                            int ind = neighborsSize/ sizeof(float) - 6; /// get the appropriate index in order to add the center point
                            vec center = sNeighbors.getCenter();
                            *(neighborsPoints+ind) = center.x; *(neighborsPoints+ind+1) = center.y; *(neighborsPoints+ind+2) = center.z;
                            *(neighborsPoints+ind+3) = 0.0f; *(neighborsPoints+ind+4) = 1.0f; *(neighborsPoints+ind+5) = 0.0f;
                #endif
                */


        int size = (trainData.size() + 1) * dimensions * 2; /// +1 refers to trainCenter, *2 refers to colours
        auto* sphericalData = new float[size];

        int ind{0};
        for (auto & p : trainData) {
            *(sphericalData+ind) = p.x;
            *(sphericalData+ind+1) = p.y;
            *(sphericalData+ind+2) = p.z;
            *(sphericalData+ind+3) = 1.0f;
            *(sphericalData+ind+4) = 0.0f;
            *(sphericalData+ind+5) = 0.0f;
            ind += 6;
        }

        /// add center
        *(sphericalData+ind) = trainCenter.x;
        *(sphericalData+ind+1) = trainCenter.y;
        *(sphericalData+ind+2) = trainCenter.z;
        *(sphericalData+ind+3) = 0.0f;
        *(sphericalData+ind+4) = 1.0f;
        *(sphericalData+ind+5) = 0.0f;

        int sphericalSize = size * sizeof(float);

        VertexArray vaSpherical;
        VertexBuffer vbSpherical(sphericalData, sphericalSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutSpherical;
        layoutSpherical.Push(3); /// the first attribute (position) has (count) floats
        layoutSpherical.Push(3); /// the first attribute (position) has (count) floats
        vaSpherical.AddVertexBuffer(vbSpherical, layoutSpherical);

        shaderSpherical.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderSpherical.SetMat4("projection", projection);
        /// camera/view transformation
        shaderSpherical.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderSpherical.SetMat4("model", model);
        renderer.Draw(vaSpherical, shaderSpherical, sphericalSize/sizeof(float), GL_POINTS); /// number of instances

        delete[] sphericalData;
#endif

#if DRAW_RINGS == 1
        int counterRings{0};
            for (unsigned int i=0; i<rings; i++)
//            for (unsigned int i=17; i<18; i++)
                counterRings += ringIndices[i].size();

            auto ringPoints = new float[counterRings * 2 * dimensions];
            int ringPointsSize = counterRings * 2 * dimensions * sizeof(float);
            int indexRings, idRings{0};
            for (unsigned int i=0; i<rings; i++) {
//            for (unsigned int i=17; i<18; i++) {

                for (unsigned j=0; j<ringIndices[i].size(); j++) {

//                    std::cout << ringIndices[i].size() << std::endl;

//                    indexRings = ringIndices[i][j].first;
                    indexRings = std::get<0>(ringIndices[i][j]); /// extract id from tuple
                    *(ringPoints+idRings) = totalGroundV1[indexRings].x;
                    *(ringPoints+idRings+1) = totalGroundV1[indexRings].y;
                    *(ringPoints+idRings+2) = totalGroundV1[indexRings].z;
                    *(ringPoints+idRings+3) = ringsCols[i].x;
                    *(ringPoints+idRings+4) = ringsCols[i].y;
                    *(ringPoints+idRings+5) = ringsCols[i].z;
                    idRings += 6;
                }
            }

            VertexArray vaAllRingPoints;
            VertexBuffer vbAllRingPoints(ringPoints, ringPointsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutVertexAllRingPoints;
            layoutVertexAllRingPoints.Push(3); /// the first attribute (position) has (count) floats
            layoutVertexAllRingPoints.Push(3); /// the first attribute (position) has (count) floats
            vaAllRingPoints.AddVertexBuffer(vbAllRingPoints, layoutVertexAllRingPoints);
            renderer.Draw(vaAllRingPoints, shader, ringPointsSize/sizeof(float), GL_POINTS); /// number of instances /// last one

            delete[] ringPoints;
#endif

#if DRAW_ORIGINAL_POINTS == 1
        auto *pointsLidarTopFinal = new float[points.size() * dimensions];
        int index{0};
        for (auto &p : points) {
            *(pointsLidarTopFinal+index) = p.x;
            *(pointsLidarTopFinal+index+1) = p.y;
            *(pointsLidarTopFinal+index+2) = p.z;
            index+=3;
        }
        int pointsSizeLidarTopFinal = points.size() * dimensions * sizeof(float);

        VertexArray vaAllPoints;
        VertexBuffer vbAllPoints(pointsLidarTopFinal, pointsSizeLidarTopFinal); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutVertexAllPoints;
        layoutVertexAllPoints.Push(3); /// the first attribute (position) has (count) floats
        vaAllPoints.AddVertexBuffer(vbAllPoints, layoutVertexAllPoints);

        shaderOriginal.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderOriginal.SetMat4("projection", projection);
        /// camera/view transformation
        shaderOriginal.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderOriginal.SetMat4("model", model);
        renderer.Draw(vaAllPoints, shaderOriginal, pointsSizeLidarTopFinal/sizeof(float), GL_POINTS); /// number of instances /// last one
        delete[] pointsLidarTopFinal;
#endif

#if DRAW_CURB == 1
//        glPointSize(5.0);

        int ind{0};
            auto* curb = new float[curbPoints.size() * dimensions];
            ind = 0;

            for (auto & curbPoint : curbPoints) {
                *(curb+ind) = curbPoint.x; *(curb+ind+1) = curbPoint.y; *(curb+ind+2) = curbPoint.z;
                ind += 3;
            }
            int curbSize = curbPoints.size() * dimensions * sizeof(float);

            VertexArray vaCurb;
            VertexBuffer vbCurb(curb, curbSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutCurb;
            layoutCurb.Push(3); /// the first attribute (position) has (count) floats
            //        layoutCurb.Push(3); /// the first attribute (position) has (count) floats
            vaCurb.AddVertexBuffer(vbCurb, layoutCurb);

            shaderCurb.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderCurb.SetMat4("projection", projection);
            /// camera/view transformation
            shaderCurb.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderCurb.SetMat4("model", model);
            renderer.Draw(vaCurb, shaderCurb, curbSize/sizeof(float), GL_POINTS); /// number of instances
            delete[] curb;
#endif
#if DRAW_ORIENTATION_VEC == 1
        auto* orien = new float[frames * dimensions];
            ind = 0;

            for (unsigned int i=0; i<frames; i++) {
                *(orien+ind) = allEgoOrien[i].x + allEgoPos[i].x;
                *(orien+ind+1) = allEgoOrien[i].y + allEgoPos[i].y;
                *(orien+ind+2) = allEgoOrien[i].z + allEgoPos[i].z;
                ind += 3;
            }
            int orienSize = frames * dimensions * sizeof(float);

            VertexArray vaOrien;
            VertexBuffer vbOrien(orien, orienSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutOrien;
            layoutOrien.Push(3); /// the first attribute (position) has (count) floats
            vaOrien.AddVertexBuffer(vbOrien, layoutOrien);

            shaderOrien.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderOrien.SetMat4("projection", projection);
            /// camera/view transformation
            shaderOrien.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderOrien.SetMat4("model", model);
            renderer.Draw(vaOrien, shaderOrien, orienSize/sizeof(float), GL_LINES); /// number of instances

            delete[] orien;
#endif

#if DRAW_ROAD_LINE == 1
        auto* roadline = new float[roadLinesUp.size() * dimensions];
            int ind = 0;
            for (auto & roadLine : roadLinesUp) {
                *(roadline+ind) = roadLine.x; *(roadline+ind+1) = roadLine.y; *(roadline+ind+2) = roadLine.z;
                ind += 3;
            }
            int roadlineSize = roadLinesUp.size() * dimensions * sizeof(float);

            VertexArray vaRoadLine;
            VertexBuffer vbRoadLine(roadline, roadlineSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutRoadLine;
            layoutRoadLine.Push(3); /// the first attribute (position) has (count) floats
            vaRoadLine.AddVertexBuffer(vbRoadLine, layoutRoadLine);

            shaderRoadLine.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderRoadLine.SetMat4("projection", projection);
            /// camera/view transformation
            shaderRoadLine.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderRoadLine.SetMat4("model", model);
            renderer.Draw(vaRoadLine, shaderRoadLine, roadlineSize/sizeof(float), GL_LINES); /// number of instances

            delete[] roadline;




            auto* roadline2 = new float[roadLinesBelow.size() * dimensions];
            int ind2 = 0;
            for (auto & roadLine : roadLinesBelow) {
                *(roadline2+ind2) = roadLine.x; *(roadline2+ind2+1) = roadLine.y; *(roadline2+ind2+2) = roadLine.z;
                ind2 += 3;
            }
            int roadlineSize2 = roadLinesBelow.size() * dimensions * sizeof(float);

            VertexArray vaRoadLine2;
            VertexBuffer vbRoadLine2(roadline2, roadlineSize2); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutRoadLine2;
            layoutRoadLine2.Push(3); /// the first attribute (position) has (count) floats
            vaRoadLine2.AddVertexBuffer(vbRoadLine2, layoutRoadLine2);

            shaderRoadLine.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderRoadLine.SetMat4("projection", projection);
            /// camera/view transformation
            shaderRoadLine.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderRoadLine.SetMat4("model", model);
            renderer.Draw(vaRoadLine2, shaderRoadLine, roadlineSize2/sizeof(float), GL_LINES); /// number of instances

            delete[] roadline2;
#endif

#if RING_OUTLIERS == 1
        auto* outlierPoints = new float[outliers.size() * dimensions];
            int outlierPointsSize = outliers.size() * dimensions * sizeof(float);
            int id = 0;
            for (auto & i : outliers) {
                *(outlierPoints+id) = i.x; *(outlierPoints+id+1) = i.y; *(outlierPoints+id+2) = i.z;
                id += 3;
            }

            VertexArray vaOutliers;
            VertexBuffer vbOutliers(outlierPoints, outlierPointsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutOutliersPoints;
            layoutOutliersPoints.Push(3); /// the first attribute (position) has (count) floats
            vaOutliers.AddVertexBuffer(vbOutliers, layoutOutliersPoints);

            shaderOutliers.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderOutliers.SetMat4("projection", projection);
            /// camera/view transformation
            shaderOutliers.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderOutliers.SetMat4("model", model);
            renderer.Draw(vaOutliers, shaderOutliers, outlierPointsSize/sizeof(float), GL_POINTS); /// number of instances

            delete[] outlierPoints;
#endif

#if DRAW_TEST == 1
        auto* testPoints = new float[debug.size() * dimensions];
        int testPointsSize = debug.size() * dimensions * sizeof(float);

        int id = 0;

        for (unsigned int i=0; i<debug.size(); i+=2) {

            *(testPoints+id) = debug[i].x; *(testPoints+id+1) = debug[i].y; *(testPoints+id+2) = debug[i].z;
            *(testPoints+id+3) = debug[i+1].x; *(testPoints+id+4) = debug[i+1].y; *(testPoints+id+5) = debug[i+1].z;
            id += 6;

        }

        VertexArray vaTest;
        VertexBuffer vbTest(testPoints, testPointsSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutTestPoints;
        layoutTestPoints.Push(3); /// the first attribute (position) has (count) floats
        layoutTestPoints.Push(3); /// the first attribute (position) has (count) floats
        vaTest.AddVertexBuffer(vbTest, layoutTestPoints);

        shaderTest.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderTest.SetMat4("projection", projection);
        /// camera/view transformation
        shaderTest.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderTest.SetMat4("model", model);
        renderer.Draw(vaTest, shaderTest, testPointsSize/sizeof(float), GL_POINTS); /// number of instances
        //        renderer.Draw(vaTest, shaderTest, testPointsSize/sizeof(float), GL_LINES); /// number of instances

        delete[] testPoints;
#endif

#if DRAW_VEHICLES
        auto* v = new float[vehicle.size() * dimensions];
        int vSize = vehicle.size() * dimensions * sizeof(float);

        int id_v = 0;

        for (unsigned int i=0; i<vehicle.size(); i++) {
            *(v+id_v) = vehicle[i].x; *(v+id_v+1) = vehicle[i].y; *(v+id_v+2) = vehicle[i].z;
            id_v += 3;
        }

        VertexArray va_veh;
        VertexBuffer vb_veh(v, vSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutv;
        layoutv.Push(3); /// the first attribute (position) has (count) floats
        va_veh.AddVertexBuffer(vb_veh, layoutv);

        shaderVehicles.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderVehicles.SetMat4("projection", projection);
        /// camera/view transformation
        shaderVehicles.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderVehicles.SetMat4("model", model);
        renderer.Draw(va_veh, shaderVehicles, vSize/sizeof(float), GL_POINTS); /// number of instances

        delete[] v;
#endif

#if DRAW_WALKERS
        auto* w = new float[walker.size() * dimensions];
        int wSize = walker.size() * dimensions * sizeof(float);

        int id = 0;

        for (unsigned int i=0; i<walker.size(); i++) {
            *(w+id) = walker[i].x; *(w+id+1) = walker[i].y; *(w+id+2) = walker[i].z;
            id += 3;
        }

        VertexArray va_w;
        VertexBuffer vb_w(w, wSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutw;
        layoutw.Push(3); /// the first attribute (position) has (count) floats
        va_w.AddVertexBuffer(vb_w, layoutw);

        shaderWalkers.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderWalkers.SetMat4("projection", projection);
        /// camera/view transformation
        shaderWalkers.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderWalkers.SetMat4("model", model);
        renderer.Draw(va_w, shaderWalkers, wSize/sizeof(float), GL_POINTS); /// number of instances

        delete[] w;
#endif

#if DRAW_ROAD_CARLA
        auto* r = new float[road.size() * dimensions];
        int rSize = road.size() * dimensions * sizeof(float);

        int id = 0;
//        id = 0;

        for (unsigned int i=0; i<road.size(); i++) {
            *(r+id) = road[i].x; *(r+id+1) = road[i].y; *(r+id+2) = road[i].z;
            id += 3;
        }

        VertexArray va_r;
        VertexBuffer vb_r(r, rSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutr;
        layoutr.Push(3); /// the first attribute (position) has (count) floats
        va_r.AddVertexBuffer(vb_r, layoutr);

        shaderRoadCarla.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderRoadCarla.SetMat4("projection", projection);
        /// camera/view transformation
        shaderRoadCarla.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderRoadCarla.SetMat4("model", model);
        renderer.Draw(va_r, shaderRoadCarla, rSize/sizeof(float), GL_POINTS); /// number of instances

        delete[] r;
#endif

#if DRAW_BOXES == 1
        int boxDim{3};
        /// get bounding boxes
        BoxArray boxes;
        dh_lyft.getBoundingBoxes(frame_id,boxes);
        int NumBoxes = boxes.size();

        /// transform corners of bounding boxes to global coordinates (into the ego vehicle frame)
        dh_lyft.TransformToGlobalCoord(boxes, vec(egoPos.x, egoPos.y, egoPos.z));

        int CornersPerBox{8}, indCorn{0}, indColCorn{0};
        int cornersSize = NumBoxes * CornersPerBox * boxDim * sizeof(float);
        auto *corners = new float[cornersSize/sizeof(float)];
        /// corners are points without color and coloredCorners are points with color where for each 3d points, we add an
        /// r,g,b value. So the new size is twice bigger than the previous
        int coloredCornersSize = 2*cornersSize;
        auto *coloredCorners = new float[coloredCornersSize/sizeof(float)];

        for (unsigned int i=0; i<NumBoxes; i++) {
            for (auto & c : boxes[i].corners) {
                *(corners+indCorn) = c.x; *(corners+indCorn+1) = c.y; *(corners+indCorn+2) = c.z;
                indCorn += 3;
                *(coloredCorners+indColCorn) = c.x; *(coloredCorners+indColCorn+1) = c.y; *(coloredCorners+indColCorn+2) = c.z;
                *(coloredCorners+indColCorn+3) = boxes[i].col.x;
                *(coloredCorners+indColCorn+4) = boxes[i].col.y;
                *(coloredCorners+indColCorn+5) = boxes[i].col.z;
                indColCorn += 6;
            }
        }

        int LinesPerBox{12}, PointsPerLine{2};
        int PointsForAllLines = NumBoxes * LinesPerBox * PointsPerLine;
        unsigned int indices[PointsForAllLines];
        renderer.CreateBoxIndices(indices, NumBoxes);
        IndexBuffer ib(indices, PointsForAllLines);

        VertexArray vaBoxes;
        VertexBuffer vbBoxes(coloredCorners, coloredCornersSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutVertexBoxes;
        layoutVertexBoxes.Push(3); /// the first attribute (position) has (count) floats
        layoutVertexBoxes.Push(3); /// the first attribute (position) has (count) floats
        vaBoxes.AddVertexBuffer(vbBoxes, layoutVertexBoxes);

        /// ---------------------------------------------draw lines-----------------------------------------------------
        shaderBox.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderBox.SetMat4("projection", projection);
        /// camera/view transformation
        shaderBox.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderBox.SetMat4("model", model);
        renderer.Draw(vaBoxes, ib, shaderBox, GL_LINES); /// number of instances
        /// ------------------------------------------------------------------------------------------------------------

        delete[] corners;
        delete[] coloredCorners;
#endif

#if DRAW_NEIGHBORS == 1
        VertexArray vaNeighbors;
            VertexBuffer vbNeighbors(neighborsPoints, neighborsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutVertexNeighbors;
            layoutVertexNeighbors.Push(3); /// the first attribute (position) has (count) floats
            layoutVertexNeighbors.Push(3); /// the first attribute (position) has (count) floats
            vaNeighbors.AddVertexBuffer(vbNeighbors, layoutVertexNeighbors);

            shaderNeighbors.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderNeighbors.SetMat4("projection", projection);
            /// camera/view transformation
            shaderNeighbors.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderNeighbors.SetMat4("model", model);
            renderer.Draw(vaNeighbors, shaderNeighbors, neighborsSize/sizeof(float), GL_POINTS); /// number of instances

            delete[] neighborsPoints;
#endif

#if DRAW_PROJECTIONS == 1
        auto projectionPoints = new float[projections.size() * dimensions];
            int projectionPointsSize = projections.size() * dimensions * sizeof(float);
            int idProj{0};
            for (unsigned int i=0; i<projections.size(); i++) {
                *(projectionPoints+idProj) = projections[i].x;
                *(projectionPoints+idProj+1) = projections[i].y;
                *(projectionPoints+idProj+2) = projections[i].z;
                idProj += 3;
            }

            VertexArray vaProjections;
            VertexBuffer vbProjections(projectionPoints, projectionPointsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutProjections;
            layoutProjections.Push(3); /// the first attribute (position) has (count) floats
            vaProjections.AddVertexBuffer(vbProjections, layoutProjections);

            shaderProjection.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderProjection.SetMat4("projection", projection);
            /// camera/view transformation
            shaderProjection.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderProjection.SetMat4("model", model);
            renderer.Draw(vaProjections, shaderProjection, projectionPointsSize/sizeof(float), GL_POINTS); /// number of instances

            delete[] projectionPoints;
#endif

#if DRAW_GROUND_V1 == 1
        auto groundV1Points = new float[totalGroundV1.size() * dimensions];
            int groundV1PointsSize = totalGroundV1.size() * dimensions * sizeof(float);
            int idGround{0};
            for (unsigned int i=0; i<totalGroundV1.size(); i++) {
                *(groundV1Points+idGround) = totalGroundV1[i].x;
                *(groundV1Points+idGround+1) = totalGroundV1[i].y;
                *(groundV1Points+idGround+2) = totalGroundV1[i].z;
                idGround += 3;
            }

//        auto groundV1Points = new float[totalGroundV2.size() * dimensions];
//            int groundV1PointsSize = totalGroundV2.size() * dimensions * sizeof(float);
//            int idGround{0};
//            for (unsigned int i=0; i<totalGroundV2.size(); i++) {
//                *(groundV1Points+idGround) = totalGroundV2[i].x;
//                *(groundV1Points+idGround+1) = totalGroundV2[i].y;
//                *(groundV1Points+idGround+2) = totalGroundV2[i].z;
//                idGround += 3;
//            }

            VertexArray vaGroundV1;
            VertexBuffer vbGroundV1(groundV1Points, groundV1PointsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutGroundV1;
            layoutGroundV1.Push(3); /// the first attribute (position) has (count) floats
            vaGroundV1.AddVertexBuffer(vbGroundV1, layoutGroundV1);

            shaderGround.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderGround.SetMat4("projection", projection);
            /// camera/view transformation
            shaderGround.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderGround.SetMat4("model", model);
            renderer.Draw(vaGroundV1, shaderGround, groundV1PointsSize/sizeof(float), GL_POINTS); /// number of instances

            delete[] groundV1Points;
#endif

#if DRAW_NON_GROUND_V1 == 1
//        auto nonGroundV1Points = new float[totalNoGroundV1.size() * dimensions];
//            int nonGroundV1PointsSize = totalNoGroundV1.size() * dimensions * sizeof(float);
//            int idNonGround{0};
//            for (unsigned int i=0; i<totalNoGroundV1.size(); i++) {
//                *(nonGroundV1Points+idNonGround) = totalNoGroundV1[i].x;
//                *(nonGroundV1Points+idNonGround+1) = totalNoGroundV1[i].y;
//                *(nonGroundV1Points+idNonGround+2) = totalNoGroundV1[i].z;
//                idNonGround += 3;
//            }

            auto nonGroundV1Points = new float[totalNoGroundV2.size() * dimensions];
            int nonGroundV1PointsSize = totalNoGroundV2.size() * dimensions * sizeof(float);
            int idNonGround{0};
            for (unsigned int i=0; i<totalNoGroundV2.size(); i++) {
                *(nonGroundV1Points+idNonGround) = totalNoGroundV2[i].x;
                *(nonGroundV1Points+idNonGround+1) = totalNoGroundV2[i].y;
                *(nonGroundV1Points+idNonGround+2) = totalNoGroundV2[i].z;
                idNonGround += 3;
            }

            VertexArray vaNonGroundV1;
            VertexBuffer vbNonGroundV1(nonGroundV1Points, nonGroundV1PointsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutNonGroundV1;
            layoutNonGroundV1.Push(3); /// the first attribute (position) has (count) floats
            vaNonGroundV1.AddVertexBuffer(vbNonGroundV1, layoutNonGroundV1);

            shaderNonGround.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderNonGround.SetMat4("projection", projection);
            /// camera/view transformation
            shaderNonGround.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderNonGround.SetMat4("model", model);
            renderer.Draw(vaNonGroundV1, shaderNonGround, nonGroundV1PointsSize/sizeof(float), GL_POINTS); /// number of instances

            delete[] nonGroundV1Points;
#endif

#if DRAW_SEGMENTS_V1 == 1
        auto allSegmentsV1Points = new float[allSegmentsV1.size() * 2 * dimensions];
            int allSegmentsV1PointsSize = allSegmentsV1.size() * 2 * dimensions * sizeof(float);
            int idSegments{0}, currentCol{0}, currentSize;
            currentSize = lengthEachSegV1[currentCol];

            for (unsigned int i=0; i<allSegmentsV1.size(); i++) {
                *(allSegmentsV1Points+idSegments) = allSegmentsV1[i].x;
                *(allSegmentsV1Points+idSegments+1) = allSegmentsV1[i].y;
                *(allSegmentsV1Points+idSegments+2) = allSegmentsV1[i].z;

                if (i > currentSize) {
                    if (currentCol != lengthEachSegV1.size()) {
                        currentCol++;
                        currentSize += lengthEachSegV1[currentCol];
                    }
                }

                *(allSegmentsV1Points+idSegments+3) = segCols[currentCol].x;
                *(allSegmentsV1Points+idSegments+4) = segCols[currentCol].y;
                *(allSegmentsV1Points+idSegments+5) = segCols[currentCol].z;

                idSegments += 6;
            }

            VertexArray vaAllSegmentsV1;
            VertexBuffer vbAllSegmentsV1(allSegmentsV1Points, allSegmentsV1PointsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutAllSegmentsV1;
            layoutAllSegmentsV1.Push(3); /// the first attribute (position) has (count) floats
            layoutAllSegmentsV1.Push(3); /// the first attribute (position) has (count) floats
            vaAllSegmentsV1.AddVertexBuffer(vbAllSegmentsV1, layoutAllSegmentsV1);

            shaderSegments.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderSegments.SetMat4("projection", projection);
            /// camera/view transformation
            shaderSegments.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderSegments.SetMat4("model", model);
            renderer.Draw(vaAllSegmentsV1, shaderSegments, allSegmentsV1PointsSize/sizeof(float), GL_POINTS); /// number of instances

            delete[] allSegmentsV1Points;
#endif

#if DRAW_SEGMENTS_V2 == 1
        auto allSegmentsV2Points = new float[allSegmentsV2->points.size() * 2 * dimensions];
            int allSegmentsV2PointsSize = allSegmentsV2->points.size() * 2 * dimensions * sizeof(float);
            int idSegments{0}, currentCol{0}, currentSize;
            currentSize = lengthEachSeg[currentCol];

            for (unsigned int i=0; i<allSegmentsV2->points.size(); i++) {
                *(allSegmentsV2Points+idSegments) = allSegmentsV2->points[i].x;
                *(allSegmentsV2Points+idSegments+1) = allSegmentsV2->points[i].y;
                *(allSegmentsV2Points+idSegments+2) = allSegmentsV2->points[i].z;

                if (i > currentSize) {
                    if (currentCol != lengthEachSeg.size()) {
                        currentCol++;
                        currentSize += lengthEachSeg[currentCol];
                    }
                }

                *(allSegmentsV2Points+idSegments+3) = segCols[currentCol].x;
                *(allSegmentsV2Points+idSegments+4) = segCols[currentCol].y;
                *(allSegmentsV2Points+idSegments+5) = segCols[currentCol].z;

                idSegments += 6;
            }

            VertexArray vaAllSegmentsV2;
            VertexBuffer vbAllSegmentsV2(allSegmentsV2Points, allSegmentsV2PointsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutAllSegmentsV2;
            layoutAllSegmentsV2.Push(3); /// the first attribute (position) has (count) floats
            layoutAllSegmentsV2.Push(3); /// the first attribute (position) has (count) floats
            vaAllSegmentsV2.AddVertexBuffer(vbAllSegmentsV2, layoutAllSegmentsV2);

            shaderSegments.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderSegments.SetMat4("projection", projection);
            /// camera/view transformation
            shaderSegments.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderSegments.SetMat4("model", model);
            renderer.Draw(vaAllSegmentsV2, shaderSegments, allSegmentsV2PointsSize/sizeof(float), GL_POINTS); /// number of instances

            delete[] allSegmentsV2Points;
#endif

#if DRAW_ROAD == 1
        int RoadInd = 0;
        auto* roadPoints = new float[totalGroundV2.size() * dimensions];
        for (auto & i : totalGroundV2) {
            *(roadPoints+RoadInd) = i.x; *(roadPoints+RoadInd+1) = i.y; *(roadPoints+RoadInd+2) = i.z;
            RoadInd += 3;
        }
        int roadPointsSize = totalGroundV2.size() * dimensions * sizeof(float);

        VertexArray vaRoad;
        VertexBuffer vbRoad(roadPoints, roadPointsSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutRoad;
        layoutRoad.Push(3); /// the first attribute (position) has (count) floats
        vaRoad.AddVertexBuffer(vbRoad, layoutRoad);

        shaderRoad.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderRoad.SetMat4("projection", projection);
        /// camera/view transformation
        shaderRoad.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderRoad.SetMat4("model", model);
        renderer.Draw(vaRoad, shaderRoad, roadPointsSize/sizeof(float), GL_POINTS); /// number of instances

        delete[] roadPoints;
#endif

#if ROAD_POINTS_VECTOR == 1
        int testInd = 0;
            auto* testPoints2 = new float[roadPointsVector.size() * dimensions];
            for (auto & i : roadPointsVector) {
                *(testPoints2+testInd) = i.x;
                *(testPoints2+testInd+1) = i.y;
                *(testPoints2+testInd+2) = i.z;
                testInd += 3;
            }
            int testPoints2Size = roadPointsVector.size() * dimensions * sizeof(float);

            VertexArray vaTest2;
            VertexBuffer vbTest2(testPoints2, testPoints2Size); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutTest;
            layoutTest.Push(3); /// the first attribute (position) has (count) floats
            vaTest2.AddVertexBuffer(vbTest2, layoutTest);

            shaderRoadPoints.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderRoadPoints.SetMat4("projection", projection);
            /// camera/view transformation
            shaderRoadPoints.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderRoadPoints.SetMat4("model", model);
            renderer.Draw(vaTest2, shaderRoadPoints, testPoints2Size/sizeof(float), GL_POINTS); /// number of instances

            delete[] testPoints2;
#endif

#if DRAW_PLANE == 1
        float u = 10.0f, v = 10.0f;
            math::vec refOrigin{0.0f, 0.0f, 0.0f};
            /// modelPlaneV1
//            math::vec p0(planeDraw.Point(-u, -v, refOrigin));
//            math::vec p1(planeDraw.Point(-u, v, refOrigin));
//            math::vec p2(planeDraw.Point(u, -v, refOrigin));
//            math::vec p3(planeDraw.Point(u, v, refOrigin));

            math::vec p0(planeDraw.Point(-u, -v,  planeDraw.PointOnPlane()));
            math::vec p1(planeDraw.Point(-u, v,  planeDraw.PointOnPlane()));
            math::vec p2(planeDraw.Point(u, -v,  planeDraw.PointOnPlane()));
            math::vec p3(planeDraw.Point(u, v,  planeDraw.PointOnPlane()));

            auto* planePoints = new float[3*3];
            *(planePoints) = p0.x; *(planePoints+1) = p0.y; *(planePoints+2) = p0.z;
            *(planePoints+3) = p1.x; *(planePoints+4) = p1.y; *(planePoints+5) = p1.z;
            *(planePoints+6) = p2.x; *(planePoints+7) = p2.y; *(planePoints+8) = p2.z;
            int planePointsSize = 3 * 3 * sizeof(float);
            LyftDatasetHandler dh;
            dh.TransformToGlobalCoord(planePoints, 9, vec(egoPos.x, egoPos.y, egoPos.z));

            VertexArray vaPlane;
            VertexBuffer vbPlane(planePoints, planePointsSize); /// size is the amount of bytes of all instances
            VertexBufferLayout layoutPlane;
            layoutPlane.Push(3); /// the first attribute (position) has (count) floats
            vaPlane.AddVertexBuffer(vbPlane, layoutPlane);

            shaderPlane.Bind();
            /// pass projection matrix to shader (note that in this case it could change every frame)
            shaderPlane.SetMat4("projection", projection);
            /// camera/view transformation
            shaderPlane.SetMat4("view", view);
            /// calculate the model matrix for each object and pass it to shader before drawing
            shaderPlane.SetMat4("model", model);
            renderer.Draw(vaPlane, shaderPlane, planePointsSize/sizeof(float), GL_TRIANGLES); /// number of instances

            delete[] planePoints;
#endif

#if DRAW_OBJECT == 1
        auto* objectPoints = new float[clusterP.size()];
        int ind = 0;
        for (unsigned int i=0; i<clusterP.size(); i+=6) {
            *(objectPoints+ind) = clusterP[i]; *(objectPoints+ind+1) = clusterP[i+1]; *(objectPoints+ind+2) = clusterP[i+2];
            *(objectPoints+ind+3) = clusterP[i+3]; *(objectPoints+ind+4) = clusterP[i+4]; *(objectPoints+ind+5) = clusterP[i+5];
            ind += 6;
        }
        int objectPointsSize = clusterP.size() * sizeof(float);

        VertexArray vaObject;
        VertexBuffer vbObject(objectPoints, objectPointsSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutObjectPoints;
        layoutObjectPoints.Push(3); /// the first attribute (position) has (count) floats
        layoutObjectPoints.Push(3); /// the first attribute (position) has (count) floats
        vaObject.AddVertexBuffer(vbObject, layoutObjectPoints);

        shaderObject.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderObject.SetMat4("projection", projection);
        /// camera/view transformation
        shaderObject.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderObject.SetMat4("model", model);
        renderer.Draw(vaObject, shaderObject, objectPointsSize/sizeof(float), GL_POINTS); /// number of instances

        delete[] objectPoints;
#endif

#if DRAW_OBJECTS_ABOVE_ROAD == 1
        int ind_demo{0};
        auto* objects = new float[objectsAboveRoad.size() * dimensions];
        ind_demo = 0;

        for (auto & p : objectsAboveRoad) {
            *(objects+ind_demo) = p.x; *(objects+ind_demo+1) = p.y; *(objects+ind_demo+2) = p.z;
            ind_demo += 3;
        }
        int objectsSize = objectsAboveRoad.size() * dimensions * sizeof(float);

        VertexArray vaObjects;
        VertexBuffer vbObjects(objects, objectsSize); /// size is the amount of bytes of all instances
        VertexBufferLayout layoutObjects;
        layoutObjects.Push(3); /// the first attribute (position) has (count) floats
        layoutObjects.Push(3); /// the first attribute (position) has (count) floats
        vaObjects.AddVertexBuffer(vbObjects, layoutObjects);

        shaderObjects.Bind();
        /// pass projection matrix to shader (note that in this case it could change every frame)
        shaderObjects.SetMat4("projection", projection);
        /// camera/view transformation
        shaderObjects.SetMat4("view", view);
        /// calculate the model matrix for each object and pass it to shader before drawing
        shaderObjects.SetMat4("model", model);
        renderer.Draw(vaObjects, shaderObjects, objectsSize/sizeof(float), GL_POINTS); /// number of instances
        delete[] objects;
        #endif

        /// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
        /// ***************************************************************
        /// handling window
        /// ***************************************************************

    #if RECORD == 1
            } /// frames iteration
        } /// glfwWindowShouldClose
    #endif

    #if RECORD == 0 && SIMPLE_RUN == 1
        } /// glfwWindowShouldClose
    #endif

    #if RECORD == 0 && SIMPLE_RUN_WITH_CLASSIFIER_ONLY == 1
        } /// glfwWindowShouldClose
    #endif

    #if RECORD == 0 && TESTING == 1
        } /// glfwWindowShouldClose
    #endif

    #if TRAINING == 1
        } /// glfwWindowShouldClose
    #endif

    /// glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();

    return 0;
}

int init() {
    /// ****************************************************************************************************************
    ///                                     INITIALIZATION (glfw, glad)
    /// ****************************************************************************************************************
    /// glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    /// glfw window creation
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    /// glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    return 0;
}

/// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    /// make sure the viewport matches the new window dimensions; note that width and
    /// height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        //getting cursor position
        glfwGetCursorPos(window, &xpos, &ypos);
    }
}

//-------------------------------------------------------------------------------------
// TODO When test it again apply diffent ratio/parameters in different classes
// CHECK ALSO https://pdfs.semanticscholar.org/910f/24c6157861a11a4d9db206ea786e4151570c.pdf?_ga=2.103506505.872748946.1601160689-116950922.1595305501
// (FOR EXAMPLE MAYBE I NEED TO NORMALIZE SOME VALUES..)
//-------------------------------------------------------------------------------------
void setCarlaDataForTraining(VecArray& points, Box3D& box, VecArray& debug, std::stringstream& trainingData) {
    int pointsSize;
    float* data_points;
    Helpers hp;

    /// define point cloud
    pointCloudBoost cloud(new pointCloud);
    SphericalNeighborhood sNeighbors;

    float initialR, smallestRadius, ratio;
    unsigned int scalesNum;

    /// setting for vehicles
    if (box.category == "vehicle") {
        initialR = 15.0f;
        smallestRadius = 1.05f;
        ratio = 1.1f;
        scalesNum = 15;
    }

    /// setting for walkers
    if (box.category == "walker") {
        initialR = 5.0f;
        smallestRadius = 0.2f;
        ratio = 1.1f;
        scalesNum = 15;
    }

    /// setting for roads
    if (box.category == "road") {
        initialR = 30.0f;
        smallestRadius = 1.5f;
        ratio = 1.2f;
        scalesNum = 15;
    }

    pointsSize = points.size();
    data_points = new float[pointsSize];

    cloud->points.clear();
    /// resize space for point cloud
    cloud->width = pointsSize; cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    /// fill point cloud with data
    int indPoint{0};
    for (auto & point : cloud->points) {
        point.x = points[indPoint].x;
        point.y = points[indPoint].y;
        point.z = points[indPoint].z;
        indPoint++;
    }

    /// TODO must optimize these parameters => (initialR, smallestRadius, ratio, scalesNum)
    sNeighbors.init(initialR, scalesNum, smallestRadius, ratio, cloud);

    vec c = hp.getCenter(box.corners);

    sNeighbors.setCenter(c);
    sNeighbors.setCloud(cloud);

    sNeighbors.run(trainingData, catToInt[box.category], debug);
    debug.emplace_back((vec(c.x, c.y, c.z)));
    debug.emplace_back(vec(0.0f, 0.0f, 1.0f));

    delete[] data_points;
}

float diff(const vec&p1, const vec&p2) {
    return sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2) + pow(p1.z-p2.z, 2));
}

bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (squared_distance < 10000)
    {
        if (std::abs (point_a.intensity - point_b.intensity) < 8.0f)
            return (true);
        if (std::abs (point_a_normal.dot (point_b_normal)) < 0.06)
            return (true);
    }
    else
    {
        if (std::abs (point_a.intensity - point_b.intensity) < 3.0f)
            return (true);
    }
    return (false);
}

template <typename T>
T getMin(const std::vector<T>& values) {
    T min = values[0];
    for (auto &v : values)
        if (v < min)
            min = v;
    return min;
}

template <typename T>
T getMax(const std::vector<T>& values) {
    T max = values[0];
    for (auto &v : values)
        if (v > max)
            max = v;
    return max;
}

void readData(std::string &pathToData, VecArray& points) {
    /// get size of binary
    std::ifstream binaryFile2;
    std::ostringstream oss;
    oss << pathToData;
    std::string name = oss.str();

    binaryFile2.open(name, std::ios::in | std::ios::binary);
    binaryFile2.seekg(0, std::ios::end);
    int size = binaryFile2.tellg();

    std::ifstream binaryFile;
    binaryFile.seekg(0, std::ios::beg);
    binaryFile.open(name, std::ios::in | std::ios::binary);

    int index2{0}; float f;
    int num{0};
//    std::cout << "loading data...\n";
    int counter{0};
    vec p;
    while(binaryFile.tellg() < size)
    {
        binaryFile.read((char*)(&f), sizeof(float));
        p.x = f;
        binaryFile.read((char*)(&f), sizeof(float));
        p.y = f;
        binaryFile.read((char*)(&f), sizeof(float));
        p.z = f;
        points.emplace_back(p);

        /// there are x,y,z,w values
        index2 += 4;
        num += 4 * sizeof(float);
        binaryFile.seekg(num, std::ios::beg);
    }
    binaryFile.close();
}

/// Read metadata such as :
/// camera_x, camera_y, camera_z, lidar_x, lidar_y, lidar_z (once)
/// vehicle_latitude, vehicle_longitude, vehicle_altitude, vehicle moving, camera_pitch, camera_yaw, camera_roll (per frame)
void readMetaData(std::string &pathToData, vec& camera_pos, vec& lidar_pos, VecArray& vehicle_pos, VecArray& extrinsics, std::vector<bool>& vehicle_moving) {
    /// get size of binary
    std::ifstream binaryFile2;
    std::ostringstream oss;
    oss << pathToData;
    std::string name = oss.str();

    binaryFile2.open(name, std::ios::in | std::ios::binary);
    binaryFile2.seekg(0, std::ios::end);
    int size = binaryFile2.tellg();

    std::ifstream binaryFile;
    binaryFile.seekg(0, std::ios::beg);
    binaryFile.open(name, std::ios::in | std::ios::binary);

    int index2{0}; float f;
    int num{0};
//    std::cout << "loading data...\n";
    int counter{0};
    vec p;
    while(binaryFile.tellg() < size)
    {
        if (counter == 0) {
            binaryFile.read((char*)(&f), sizeof(float));
            p.x = f;
            binaryFile.read((char*)(&f), sizeof(float));
            p.y = f;
            binaryFile.read((char*)(&f), sizeof(float));
            p.z = f;
            camera_pos = p;

            binaryFile.read((char*)(&f), sizeof(float));
            p.x = f;
            binaryFile.read((char*)(&f), sizeof(float));
            p.y = f;
            binaryFile.read((char*)(&f), sizeof(float));
            p.z = f;
            lidar_pos = p;

            /// there are x,y,z,w values
            index2 += 6;
            num += 6 * sizeof(float);
            binaryFile.seekg(num, std::ios::beg);
        }
        else {
            binaryFile.read((char*)(&f), sizeof(float));
            p.x = f;
            binaryFile.read((char*)(&f), sizeof(float));
            p.y = f;
            binaryFile.read((char*)(&f), sizeof(float));
            p.z = f;
            vehicle_pos.emplace_back(p);

            binaryFile.read((char*)(&f), sizeof(float));
            vehicle_moving.emplace_back(f == 1.0f);

//            std::cout << f << std::endl;

            binaryFile.read((char*)(&f), sizeof(float));
            p.x = f;
            binaryFile.read((char*)(&f), sizeof(float));
            p.y = f; /// set this in case of training

            /// set this in case of testing
//            float tmp = f + 180.0f;
//            if (tmp > 360.0f)
//                tmp = 360.0f - f;
//            if (tmp < -360.0f)
//                tmp = 360.0f + f;
//            p.y = tmp;

            binaryFile.read((char*)(&f), sizeof(float));
            p.z = f;
            extrinsics.emplace_back(p);

            /// there are x,y,z,w values
            index2 += 7;
            num += 7 * sizeof(float);
            binaryFile.seekg(num, std::ios::beg);
        }
        counter++;
    }
    binaryFile.close();
}

void estimateRoadLines(Eigen::VectorXd &roadUp, Eigen::VectorXd& roadBelow, VecArray& curbUp, VecArray& curbBelow, VecArray &curbPoints, VecArray &roadPointsVector, VecArray &totalGroundV1, VecArray &totalNoGroundV1, vec &ego_location, VecArray& testup, VecArray& testbelow, int draw_road_line) {
    CurbDetection curbDetect;

    /// define threshold to decide whether a line is straight or not
    #if CARLA_DATA == 1
        float line_threshold{0.7f}; /// old

//        float line_threshold{0.7f}; /// test

    #endif

    #if LYFT_DATA == 1
        float line_threshold{5.0f};
    #endif

    bool isCurvedUp = curbDetect.isCurved(curbUp, line_threshold);
    if (isCurvedUp) std::cout << "curb up is curved\n";
    else std::cout << "curb up is straight\n";

    bool success{true};
    roadUp = curbDetect.getRoadEquation(isCurvedUp, curbUp, success);

    bool isCurvedBelow = curbDetect.isCurved(curbBelow, line_threshold);
    if (isCurvedBelow) std::cout << "curb below is curved\n";
    else std::cout << "curb below is straight\n";

    roadBelow = curbDetect.getRoadEquation(isCurvedBelow, curbBelow, success);

    /// update curb points excluding outliers
    curbPoints.clear();
    curbPoints.insert(curbPoints.end(), curbUp.begin(), curbUp.end());
    curbPoints.insert(curbPoints.end(), curbBelow.begin(), curbBelow.end());

    VecArray allPoints;
    for (auto &p : totalGroundV1)
        if (p.x >= ego_location.x)
            allPoints.emplace_back(p);
    for (auto &p : totalNoGroundV1)
        if (p.x >= ego_location.x)
            allPoints.emplace_back(p);

    roadPointsVector.clear();
    curbDetect.getRoadPoints(isCurvedUp, isCurvedBelow, roadUp, roadBelow, curbUp, curbBelow, allPoints, roadPointsVector);
}

void saveImage(const char* filepath, GLFWwindow* w) {
    int width, height;
    glfwGetFramebufferSize(w, &width, &height);
    GLsizei nrChannels = 3;
    GLsizei stride = nrChannels * width;
    stride += (stride % 4) ? (4 - stride % 4) : 0;
    GLsizei bufferSize = stride * height;
    std::vector<char> buffer(bufferSize);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
}

void write_to_bin(VecArray& points, std::string& filename, bool flag) {
    std::fstream myFile(filename, std::ios::out | std::ios::binary);
    float intensity{100.0f};
    /// points
    if (flag == 0) {
        for (auto&p : points) {
            myFile.write ((char*)&p.x, sizeof(float));
            myFile.write ((char*)&p.y, sizeof(float));
            myFile.write ((char*)&p.z, sizeof(float));
            myFile.write ((char*)&intensity, sizeof(float));
        }
        myFile.close();
    }
    else {
        for (auto&p : points) {
            myFile.write ((char*)&p.x, sizeof(float));
            myFile.write ((char*)&p.y, sizeof(float));
            myFile.write ((char*)&p.z, sizeof(float));
        }
        myFile.close();
    }
}
