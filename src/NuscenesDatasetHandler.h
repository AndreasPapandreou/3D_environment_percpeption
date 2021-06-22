#ifndef LAB0_NUSCENESDATASETHANDLER_H
#define LAB0_NUSCENESDATASETHANDLER_H

#include <iostream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <map>
#include "DataStructures.h"

class NuscenesDatasetHandler {
private:
    std::vector<Annotation> m_annotations;
    std::vector<Sample> m_samples;
    std::vector<Scene> m_scenes;
    std::vector<my_Polygon> m_roadSegments;
    std::map<std::string, int> m_mapSamples;
    std::map<std::string, int> m_mapScenes;

public:
    void getBinaryData(const std::string& filePath, int size, float*& values);
    static int getBinarySize(const std::string& filePath);
    std::string ReadJson(const std::string& filepath);
    void ParseJson(const std::string& data);
    inline void GatherSamples(nlohmann::json& json, std::vector<unsigned int>& indices);
    inline void GatherAnnotations(nlohmann::json& json, std::vector<unsigned int>& indices, std::string& LidarPath);
    inline void GatherRoadSegments(nlohmann::json& json, std::vector<unsigned int>& indices);
    void getBoundingBoxes(const int& SampleIndex, BoxArray& boxes, const std::string& category = "");
    static void TransformToGlobalCoord(BoxArray& boxes, const vec& position);
    static void TransformToGlobalCoord(Box3D& box, const vec& position);
    static void TransformToGlobalCoord(float* data, unsigned int length, const vec& position);
    int getSceneId(const std::string& token);
    int getSampleId(const int& SceneIndex, const std::string& SampleToken);
    const vec getAnnotationColor(const std::string& annotation);
    void showResults();

    std::vector<Annotation> getAnnotations();
    std::vector<Sample>  getSamples();
    std::vector<Scene> getScenes();
    std::vector<my_Polygon> getRoadSegments();
};

#endif //LAB0_NUSCENESDATASETHANDLER_H