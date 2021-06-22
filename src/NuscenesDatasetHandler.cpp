#include "NuscenesDatasetHandler.h"
#include "DataConversions.h"

int NuscenesDatasetHandler::getBinarySize(const std::string& filePath) {
    std::ifstream binaryFile;
    binaryFile.open(filePath, std::ios::in | std::ios::binary);
    binaryFile.seekg(0, std::ios::end);
    return (int) binaryFile.tellg();
}

void NuscenesDatasetHandler::getBinaryData(const std::string& filePath, int size, float*& values) {
    std::ifstream binaryFile;
    binaryFile.seekg(0, std::ios::beg);
    binaryFile.open(filePath, std::ios::in | std::ios::binary);

    int index{0}; float f;
    int num{0};
    while(binaryFile.tellg() < size)
    {
        binaryFile.read((char*)(&f), sizeof(float));
        *(values+index) = f;

        binaryFile.read((char*)(&f), sizeof(float));
        *(values+index+1) = f;

        binaryFile.read((char*)(&f), sizeof(float));
        *(values+index+2) = f;

        index += 3;
        // TODO check what are these floats in Lyft!!! (the fourth is the intensity)
        num += 5 * sizeof(float); /// pass the next two floats
        binaryFile.seekg(num, std::ios::beg);
    }
    binaryFile.close();
}

/// \description
/// Read json file
/// \param The path of json
/// \return String which stores the content of json
std::string NuscenesDatasetHandler::ReadJson(const std::string& filepath) {
    std::cout << "in readjson" << std::endl;
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
void NuscenesDatasetHandler::ParseJson(const std::string& data) {
//    std::cout << "in ParseJson" << std::endl;

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
        nlohmann::json Samples = el.value();
        /// extract all the samples for each scene
        GatherSamples(Samples, SampleIndices);
        /// store the scenes
//        std::cout << "add to scenes" << std::endl;
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
inline void NuscenesDatasetHandler::GatherSamples(nlohmann::json& json, std::vector<unsigned int>& SampleIndices) {
//    std::cout << "in GatherSamples" << std::endl;

    /// This vector keeps some indices of vector of Annotations
    std::vector<unsigned int> AnnIndices, RoadIndices;

    /// get the value of key = samples
    json = json["samples"];


    /// iterate through all samples (in json form) where the key represents the sample id and the values stores all
    /// of its metadata (annotations or lidar up to now..)
    std::string SampleId, LidarPath;
    for (auto& el : json.items()) {
        AnnIndices.clear(); RoadIndices.clear();
        SampleId = el.key();
        /// store the lidar data
        if (el.value().find("lidar") != el.value().end()) {
            /// el.value() refers to json data, so below we get the value of key = lidar
            LidarPath = el.value()["lidar"];
        }
        /// store the annotations
        if (el.value().find("annotations") != el.value().end()) {
            nlohmann::json Annotations = el.value();
            GatherAnnotations(Annotations, AnnIndices, LidarPath);
        }
        /// store the polygons
//        std::cout << "before road segment" << std::endl;
//        if (el.value().find("road_segment") != el.value().end()) {
//            nlohmann::json RoadSegments = el.value();
//            GatherRoadSegments(RoadSegments, RoadIndices);
//        }
//        std::cout << "add to sample" << std::endl;
        /// store the samples
        m_samples.emplace_back(SampleId, LidarPath, AnnIndices, RoadIndices);
        m_mapSamples.insert((std::pair<std::string, int>(SampleId, m_samples.size()-1)));
        SampleIndices.emplace_back(m_samples.size()-1);
    }
}

/// \description
/// Store all annotations of lyft dataset =>
/// Parse each annotation (in json form) and store its content to some data structures. Start iterating each annotation
/// and store each one of them.
/// \param Annotation in json form and a vector of unsigned int
/// \return void
inline void NuscenesDatasetHandler::GatherAnnotations(nlohmann::json& json, std::vector<unsigned int>& AnnIndices, std::string& LidarPath) {

//    std::cout << "in GatherAnnotations" << std::endl;

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
        m_annotations.emplace_back(AnnotationId, box, LidarPath);
        AnnIndices.emplace_back(m_annotations.size()-1);
    }
}

/// \description
/// \param
/// \return
inline void NuscenesDatasetHandler::GatherRoadSegments(nlohmann::json& json, std::vector<unsigned int>& RoadIndices) {
//    std::cout << "in GatherRoadSegments" << std::endl;

    /// get the value of key = road_segment
    json = json["road_segment"];
    my_Polygon pol;
    pol.nodes = convertToVecArray(json.get<std::vector<float>>());
    m_roadSegments.emplace_back(pol);
    RoadIndices.emplace_back(m_roadSegments.size()-1);
}

std::vector<Sample>  NuscenesDatasetHandler::getSamples() {
    return m_samples;
}

std::vector<Scene> NuscenesDatasetHandler::getScenes() {
    return m_scenes;
}

std::vector<Annotation> NuscenesDatasetHandler::getAnnotations() {
    return m_annotations;
}

std::vector<my_Polygon> NuscenesDatasetHandler::getRoadSegments() {
    return m_roadSegments;
}

/// \description
/// Return all bounding boxes which refer to a specific sample of a scene.
/// \param The id of scene as string, the id of sample as string and a vector with the bounding boxes
/// \return void
void NuscenesDatasetHandler::getBoundingBoxes(const int& SampleIndex, BoxArray& boxes, const std::string& category) {
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

void NuscenesDatasetHandler::TransformToGlobalCoord(BoxArray& boxes, const vec& pos) {
    for (auto & box : boxes) {
        /// modify 3d points, so transform each dimension separately
        /// add to each corner the position of ego vehicle
        for (auto & corner : box.corners) {
            corner.x += pos.x; corner.y += pos.y; corner.z += pos.z;
        }
    }
}

void NuscenesDatasetHandler::TransformToGlobalCoord(Box3D& box, const vec& pos) {
    /// modify 3d points, so transform each dimension separately
    /// add to each corner the position of ego vehicle
    for (auto & corner : box.corners) {
        corner.x += pos.x; corner.y += pos.y; corner.z += pos.z;
    }
}

void NuscenesDatasetHandler::TransformToGlobalCoord(float *data, unsigned int length, const vec& position) {
    /// modify 3d points, so transform each dimension separately
    /// add to each corner the position of ego vehicle
    for (unsigned int j=0; j<length; j+=3) {
        data[j] += position.x; data[j+1] += position.y; data[j+2] += position.z;
    }
}

/// choose specific scene
int NuscenesDatasetHandler::getSceneId(const std::string& SceneToken) {
    return m_mapScenes.at(SceneToken);
}

/// choose specific sample
int NuscenesDatasetHandler::getSampleId(const int& SceneIndex, const std::string& SampleToken) {
    return m_mapSamples.at(SampleToken);
}

const vec NuscenesDatasetHandler::getAnnotationColor(const std::string& annotation) {
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

void NuscenesDatasetHandler::showResults() {
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
        std::cout << "lidarTopPath = " << m_samples[i].LidarTopPath << std::endl;
        std::cout << "LidarFrontRightPath = " << m_samples[i].LidarFrontRightPath << std::endl;
        std::cout << "LidarFrontLeftPath = " << m_samples[i].LidarFrontLeftPath << std::endl;
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