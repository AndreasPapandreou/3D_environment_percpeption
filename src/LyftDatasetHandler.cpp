#include "LyftDatasetHandler.h"
#include "DataConversions.h"

int LyftDatasetHandler::getBinarySize(const std::string& filePath) {
    std::ifstream binaryFile;
    binaryFile.open(filePath, std::ios::in | std::ios::binary);
    binaryFile.seekg(0, std::ios::end);
    return (int) binaryFile.tellg();
}

void LyftDatasetHandler::getBinaryData(const std::string& filePath, int size, float*& values, VecArray& points) {
    std::ifstream binaryFile;
    binaryFile.seekg(0, std::ios::beg);
    binaryFile.open(filePath, std::ios::in | std::ios::binary);

    int index{0}; float f;
    int num{0};
    vec p;
    while(binaryFile.tellg() < size)
    {
        binaryFile.read((char*)(&f), sizeof(float));
        *(values+index) = f;
        p.x = f;

        binaryFile.read((char*)(&f), sizeof(float));
        *(values+index+1) = f;
        p.y = f;

        binaryFile.read((char*)(&f), sizeof(float));
        *(values+index+2) = f;
        p.z = f;

        points.emplace_back(p);

        index += 3;
        num += 5 * sizeof(float); /// pass the next two floats (intensity, ring index)
        binaryFile.seekg(num, std::ios::beg);
    }
    binaryFile.close();
}

/// \description
/// Read json file
/// \param The path of json
/// \return String which stores the content of json
std::string LyftDatasetHandler::ReadJson(const std::string& filepath) {
    std::string line;
    std::ifstream myfile(filepath);
    std::stringstream ss;
    if (myfile.is_open()) {
        while ( getline (myfile,line) ) {
            ss << line << '\n';
        }
        myfile.close();
    }
    else {
        std::cout << "file not opened" << std::endl;
    }
    return ss.str();
}

/// \description
/// Store all scenes of lyft dataset =>
/// Parse json file and store its content to some data structures. Start iterating each scene and store each one of
/// them. For each scene also, keep the indices of samples that exist there.
/// \param Json data as string
/// \return void
void LyftDatasetHandler::ParseJson(const std::string& data, const std::string& pathTolyftData) {
    /// This vector keeps some indices of vector of Samples
    std::vector<unsigned int> SampleIndices;
    /// parse json data
    auto json = nlohmann::json::parse(data);
    /// get the value of key = scenes
    json = json["scenes"];
    /// get the number of scenes
    unsigned int NumScenes = json.size();
    /// reserve memory for optimization purposes
    m_scenes.reserve(NumScenes);

    /// iterate through all scenes (in json form) where the key represents the scene id and the values stores all
    /// of its metadata
    std::string SceneId;
    for (auto& el : json.items()) {
        SampleIndices.clear();
        SceneId = el.key();

//        std::cout << "scene id = " << SceneId << std::endl;

        nlohmann::json Samples = el.value();
        /// extract all the samples for each scene
        GatherSamples(Samples, SampleIndices, pathTolyftData);
        /// store the scenes
        m_scenes.emplace_back(SceneId, SampleIndices);
        m_mapScenes.insert((std::pair<std::string, int>(SceneId, m_scenes.size()-1)));
    }
//    showResults();
}

/// \description
/// Store all samples of lyft dataset =>
/// Parse each sample (in json form) and store its content to some data structures. Start iterating each sample and
/// store each one of them. For each sample also, keep the indices of annotations that exist there.
/// \param Sample in json form and a vector of unsigned int
/// \return void
inline void LyftDatasetHandler::GatherSamples(nlohmann::json& json, std::vector<unsigned int>& SampleIndices, const std::string& pathTolyftData) {
    /// This vector keeps some indices of vector of Annotations
    std::vector<unsigned int> AnnIndices;

    /// get the value of key = samples
    json = json["samples"];

    /// iterate through all samples (in json form) where the key represents the sample id and the values stores all
    /// of its metadata (annotations or lidar up to now..)
    std::string SampleId, LidarTopPath, LidarFrontRightPath, LidarFrontLeftPath;
    float yaw, pitch, roll;
    VecArray lidar_top_rotation, ego_pose_rotation;

    for (auto& el : json.items()) {
        AnnIndices.clear();
        SampleId = el.key();

        /// store the lidar top data
        if (el.value().find("LIDAR_TOP") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = lidar top
            LidarTopPath = el.value()["LIDAR_TOP"];
        }
        /// store the lidar top yaw
        if (el.value().find("LIDAR_TOP_YAW") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = LIDAR_TOP_YAW
            yaw = el.value()["LIDAR_TOP_YAW"];
        }
        /// store the lidar top pitch
        if (el.value().find("LIDAR_TOP_PITCH") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = LIDAR_TOP_PITCH
            pitch = el.value()["LIDAR_TOP_PITCH"];
        }
        /// store the lidar top roll
        if (el.value().find("LIDAR_TOP_ROLL") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = LIDAR_TOP_ROLL
            roll = el.value()["LIDAR_TOP_ROLL"];
        }
        /// store the lidar top rotation [x,y,z]
        if (el.value().find("LIDAR_TOP_ROTATION") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = LIDAR_TOP_ROTATION
            lidar_top_rotation = convertToVecArray(el.value()["LIDAR_TOP_ROTATION"].get<std::vector<float>>());
        }
        /// store the ego pose rotation [x,y,z]
        if (el.value().find("EGO_POSE_ROTATION") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = EGO_POSE_ROTATION
            ego_pose_rotation = convertToVecArray(el.value()["EGO_POSE_ROTATION"].get<std::vector<float>>());
        }
        /// store the lidar front right data
        if (el.value().find("LIDAR_FRONT_RIGHT") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = lidar front right
            LidarFrontRightPath = el.value()["LIDAR_FRONT_RIGHT"];
        }
        /// store the lidar front left data
        if (el.value().find("LIDAR_FRONT_LEFT") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = lidar front left
            LidarFrontLeftPath = el.value()["LIDAR_FRONT_LEFT"];
        }
        /// store the annotations
        if (el.value().find("annotations") != el.value().end()) {
            nlohmann::json Annotations = el.value();
            GatherAnnotations(Annotations, AnnIndices, LidarTopPath);
        }

        /// gather paths of all camera images
        std::vector<std::string> PathToCameraImages;
        std::string eachCam;

        eachCam = el.value()["cam_front"];
        PathToCameraImages.emplace_back(pathTolyftData + eachCam);
        eachCam = el.value()["cam_front_left"];
        PathToCameraImages.emplace_back(pathTolyftData + eachCam);
        eachCam = el.value()["cam_front_right"];
        PathToCameraImages.emplace_back(pathTolyftData + eachCam);
        eachCam = el.value()["cam_back"];
        PathToCameraImages.emplace_back(pathTolyftData + eachCam);
        eachCam = el.value()["cam_back_left"];
        PathToCameraImages.emplace_back(pathTolyftData + eachCam);
        eachCam = el.value()["cam_back_right"];
        PathToCameraImages.emplace_back(pathTolyftData + eachCam);

//      to delete

//        eachCam = el.value()["cam_front"];
//        PathToCameraImages.emplace_back("/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/" + eachCam);
//        eachCam = el.value()["cam_front_left"];
//        PathToCameraImages.emplace_back("/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/" + eachCam);
//        eachCam = el.value()["cam_front_right"];
//        PathToCameraImages.emplace_back("/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/" + eachCam);
//        eachCam = el.value()["cam_back"];
//        PathToCameraImages.emplace_back("/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/" + eachCam);
//        eachCam = el.value()["cam_back_left"];
//        PathToCameraImages.emplace_back("/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/" + eachCam);
//        eachCam = el.value()["cam_back_right"];
//        PathToCameraImages.emplace_back("/home/andreas/Dropbox/Datasets/Lyft_dataset/nuscenes-devkit-master/python-sdk/data/sets/nuscenes/" + eachCam);

        /// store the samples
        m_samples.emplace_back(SampleId, LidarTopPath, LidarFrontRightPath, LidarFrontLeftPath, AnnIndices, PathToCameraImages, yaw, pitch, roll, lidar_top_rotation, ego_pose_rotation);
        m_mapSamplesWithId.insert((std::pair<std::string, int>(SampleId, m_samples.size()-1)));
        m_mapSamplesWithLidarTop.insert((std::pair<std::string, int>(LidarTopPath, m_samples.size()-1)));
        SampleIndices.emplace_back(m_samples.size()-1);
    }
}

/// \description
/// Store all annotations of lyft dataset =>
/// Parse each annotation (in json form) and store its content to some data structures. Start iterating each annotation
/// and store each one of them.
/// \param Annotation in json form and a vector of unsigned int
/// \return void
inline void LyftDatasetHandler::GatherAnnotations(nlohmann::json& json, std::vector<unsigned int>& AnnIndices, std::string& LidarTopPath) {
    std::string AnnotationId, category;
    VecArray corners;

    /// get the value of key = annotations
    json = json["annotations"];

    /// iterate through all annotations (in json form)
    for (auto& el : json.items()) {
        AnnotationId = el.key();

        /// store the category
        if (el.value().find("category") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = category
            category = el.value()["category"];
        }
        /// store the corners
        if (el.value().find("corners") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = corners
            el.value() = el.value()["corners"];
            corners = convertToVecArray(el.value().get<std::vector<float>>());
        }

        const vec col = getAnnotationColor(category);
        Box3D box(corners, category, col);
        m_annotations.emplace_back(AnnotationId, box, LidarTopPath);
        AnnIndices.emplace_back(m_annotations.size()-1);
    }
}

std::vector<Sample>  LyftDatasetHandler::getSamples() {
    return m_samples;
}

std::vector<Scene> LyftDatasetHandler::getScenes() {
    return m_scenes;
}

std::vector<Annotation> LyftDatasetHandler::getAnnotations() {
    return m_annotations;
}

/// \description
/// Return all bounding boxes which refer to a specific sample of a scene.
/// \param The id of scene as string, the id of sample as string and a vector with the bounding boxes
/// \return void
void LyftDatasetHandler::getBoundingBoxes(const int& SampleIndex, BoxArray& boxes, const std::string& category) {
    /// find the annotation(s)
    for (unsigned int AnnIndex : m_samples[SampleIndex].AnnIndices) {
        /// if a category has been chosen
        if (!category.empty()) {
            if (m_annotations[AnnIndex].box.category == category) {
                boxes.emplace_back(m_annotations[AnnIndex].box);
            }
        } else {
            boxes.emplace_back(m_annotations[AnnIndex].box);
        }
    }
}

void LyftDatasetHandler::TransformToGlobalCoord(BoxArray& boxes, const vec& pos) {
    for (auto & box : boxes) {
        /// modify 3d points, so transform each dimension separately
        /// add to each corner the position of ego vehicle
        for (auto & corner : box.corners) {
            corner.x += pos.x; corner.y += pos.y; corner.z += pos.z;
        }
    }
}

void LyftDatasetHandler::TransformToGlobalCoord(Box3D& box, const vec& pos) {
    /// modify 3d points, so transform each dimension separately
    /// add to each corner the position of ego vehicle
    for (auto & corner : box.corners) {
        corner.x += pos.x; corner.y += pos.y; corner.z += pos.z;
    }
}

void LyftDatasetHandler::TransformToGlobalCoord(VecArray& points, const vec& pos) {
    /// modify 3d points, so transform each dimension separately
    /// add to each corner the position of ego vehicle
    for (auto & p : points) {
        p.x += pos.x; p.y += pos.y; p.z += pos.z;
    }
}

void LyftDatasetHandler::TransformToGlobalCoord(float *data, unsigned int length, const vec& position) {
    /// modify 3d points, so transform each dimension separately
    /// add to each corner the position of ego vehicle
    for (unsigned int j=0; j<length; j+=3) {
//        std::cout << "point " << data[j] << ", " << data[j+1] << ", "<< data[j+2] << " became \n";
        data[j] += position.x; data[j+1] += position.y; data[j+2] += position.z;
//        std::cout << data[j] << ", " << data[j+1] << ", "<< data[j+2] << std::endl;
    }
}

/// choose specific scene
int LyftDatasetHandler::getSceneId(const std::string& SceneToken) {
    return m_mapScenes.at(SceneToken);
}

/// choose specific sample
int LyftDatasetHandler::getSampleIdGivenToken(const int& SceneIndex, const std::string& SampleToken) {
    return m_mapSamplesWithId.at(SampleToken);
}

/// choose specific sample
int LyftDatasetHandler::getSampleIdGivenLidarTop(const std::string& LidarTop) {
    return m_mapSamplesWithLidarTop.at(LidarTop);
}

const vec LyftDatasetHandler::getAnnotationColor(const std::string& annotation) {
    // TODO handle if annotation does not exist
    vec col{0.59f, 0.29f, 0.0f}; /// set the default color as brown
    if (annotation == "animal") col = vec(1.0, 1.0, 1.0); /// white
    if (annotation == "bicycle") col = vec(1.0, 0.02, 0.02); /// red
    if (annotation == "bus") col = vec(0.94, 1.0, 0.1); /// yellow
    if (annotation == "car") col = vec(0.17, 1.0, 0.08); /// green
    if (annotation == "emergency_vehicle") col = vec(0.09, 0.97, 1.0); /// aqua (Light Blue)
    if (annotation == "motorcycle") col = vec(0.05, 0.08, 1.0); /// blue
    if (annotation == "pedestrian") col = vec(1.0, 0.05, 0.84); /// magenta
    if (annotation == "other_vehicle") col = vec(0, 0, 0); /// black
    if (annotation == "truck") col = vec(1.0, 0.56, 0.11); /// orange
    return col;
}

void LyftDatasetHandler::showResults() {
    std::cout << "$$$$$$ RESULTS $$$$$$" << std::endl;
    std::cout << "num of m_annotations = " << m_annotations.size() << std::endl;
    std::cout << "num of m_samples = " << m_samples.size() << std::endl;
    std::cout << "num of m_scenes = " << m_scenes.size() << std::endl;
    std::cout << "$$$$$$ ANNOTATIONS $$$$$$" << std::endl;
    for (unsigned int i=0; i<m_annotations.size(); i++) {
        std::cout << "id = " << m_annotations[i].id << std::endl;
//        std::cout << "category = " << m_annotations[i].category << std::endl;
        for (unsigned int j=0; j<m_annotations[i].box.corners.size(); j++) {
            std::cout << "category = " << m_annotations[i].box.category << std::endl;
            std::cout << "corner_ " << j << " = " << m_annotations[i].box.corners[j].x << " "  <<
                      m_annotations[i].box.corners[j].y << " "  <<
                      m_annotations[i].box.corners[j].z << " "  << std::endl;
        }
    }

    std::cout << "$$$$$$ SAMPLES $$$$$$" << std::endl;
    for (unsigned int i=0; i<m_samples.size(); i++) {
        std::cout << "id = " << m_samples[i].id << std::endl;
        std::cout << "lidarPath = " << m_samples[i].LidarTopPath << std::endl;
        for (unsigned int j=0; j<m_samples[i].AnnIndices.size(); j++) {
            std::cout << "index_ " << j << " = " << m_samples[i].AnnIndices[j] << std::endl;
        }
    }

    std::cout << "$$$$$$ SCENES $$$$$$" << std::endl;
    for (unsigned int i=0; i<m_scenes.size(); i++) {
        std::cout << "id = " << m_scenes[i].id << std::endl;
        for (unsigned int j=0; j<m_scenes[i].SampleIndices.size(); j++) {
            std::cout << "index_ " << j << " = " << m_scenes[i].SampleIndices[j] << std::endl;
        }
    }
}

/// this method is based in [ G. Wang, J. Wu, R. He and S. Yang, "A Point Cloud-Based Robust Road Curb Detection and
/// Tracking Method," in IEEE Access, vol. 7, pp. 24611-24625, 2019. ]
void LyftDatasetHandler::distortionCorrection(float*& data, unsigned int& length, float& yaw, float& pitch, float& roll) {

    /// !!!!!!!!!!!!!!!!!!!!!!!!!!!1111
    /// not working properly...

    /// lidar runs in 10Hz
    float  scan_period = 0.1f;
    float azimuth = 0.2f; /// in degrees

    float ti; /// the time required for LiDAR rotation from i position until the end of the current frame in one scan line

    Eigen::Vector3f displacement_matrix, pose_matrix, old_point, new_point, yaw_pitch_roll;
    yaw_pitch_roll << yaw, pitch, roll;

    float x,y,z;

    SphericalCoordinates sc_p; /// SphericalCoordinates
    Eigen::Matrix3f rotation_x, rotation_y, rotation_z, rotation_all;

    for (unsigned int i=0; i<length; i+=3) {

        /// read coordinates
        x = *(data+i); y = *(data+i+1); z = *(data+i+2);
        old_point << x, y, z;

        /// convert to sphericalCoordinates
        sc_p = convertToSpherical(vec(x, y, z));

        /// compute the time required for LiDAR rotation from i position until the end of the current frame
        ti = scan_period * sc_p.th / 360.0f;

//        if (sc_p.th > 180.0f)
//            ti = scan_period * ( 360.0f - (sc_p.th - 180.0f)) / 360.0f;
//        else
//            ti = scan_period * ( 360.0f - (180.0f - sc_p.th)) / 360.0f;

        /// add values to displacement matrix for each axis separately
        ///old_point E (3x1) && old_point.transpose() E (1x3)
        displacement_matrix = (ti/scan_period) * old_point;

        /// add values to pose matrix for each axis separately
        pose_matrix = (ti/scan_period) * yaw_pitch_roll;

        /// calculate rotation about x axis
        rotation_x << 1, 0, 0,
                0, cos(pose_matrix[0]), -sin(pose_matrix[0]),
                0, sin(pose_matrix[0]), cos(pose_matrix[0]);

        /// calculate rotation about y axis
        rotation_y << cos(pose_matrix[1]), 0, sin(pose_matrix[1]),
                      0, 1, 0,
                      -sin(pose_matrix[1]), 0, cos(pose_matrix[1]);

        /// calculate rotation about z axis
        rotation_z << cos(pose_matrix[2]), -sin(pose_matrix[2]), 0,
                      sin(pose_matrix[2]), cos(pose_matrix[2]), 0,
                      0, 0, 1;

        /// compute the total rotation matrix
        rotation_all = rotation_z * rotation_y * rotation_x;

        /// point correction process
        new_point = rotation_all * old_point + displacement_matrix;
        *(data+i) = new_point[0];*(data+i+1) = new_point[1]; *(data+i+2) = new_point[2];
//        std::cout << "old point = " << old_point << " and new point = " << new_point << std::endl;
    }
}

SphericalCoordinates LyftDatasetHandler::convertToSpherical(const vec &point) {
    SphericalCoordinates sc;
    sc.r = float(sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2)));

    /// Inspired by https://stackoverflow.com/questions/283406/what-is-the-difference-between-atan-and-atan2-in-c/12011762#12011762
    sc.th = float(atan2(point.y, point.x) * 180.0f / M_PI); /// convert radian to degree

    if(sc.th < 0.0f)
        sc.th += 360.0f;

    sc.f = float(atan2(sqrt(point.x*point.x + point.y*point.y), point.z) * 180.0f / M_PI); /// convert radian to degree
    return sc;
}

void LyftDatasetHandler::getMapSamples(std::map<std::string, int>& mapSamples) {
    mapSamples = m_mapSamplesWithId;
}