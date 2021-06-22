#ifndef LAB0_LYFTDATASETHANDLER_H
#define LAB0_LYFTDATASETHANDLER_H

#include <iostream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <map>
#include "DataStructures.h"

class LyftDatasetHandler {
private:
    std::vector<Annotation> m_annotations;
    std::vector<Sample> m_samples;
    std::vector<Scene> m_scenes;
    std::map<std::string, int> m_mapSamplesWithId;
    std::map<std::string, int> m_mapSamplesWithLidarTop;
    std::map<std::string, int> m_mapScenes;

public:
    void getBinaryData(const std::string& filePath, int size, float*& values, VecArray& points);
    static int getBinarySize(const std::string& filePath);
    std::string ReadJson(const std::string& filepath);
    void ParseJson(const std::string& data, const std::string& pathTolyftData);
    inline void GatherSamples(nlohmann::json& json, std::vector<unsigned int>& indices, const std::string& pathTolyftData);
    inline void GatherAnnotations(nlohmann::json& json, std::vector<unsigned int>& indices, std::string& LidarPath);
    void getBoundingBoxes(const int& SampleIndex, BoxArray& boxes, const std::string& category = "");
    static void TransformToGlobalCoord(BoxArray& boxes, const vec& position);
    static void TransformToGlobalCoord(Box3D& box, const vec& position);
    static void TransformToGlobalCoord(float* data, unsigned int length, const vec& position);
    static void TransformToGlobalCoord(VecArray& points, const vec& pos);
    int getSceneId(const std::string& token);
    int getSampleIdGivenToken(const int& SceneIndex, const std::string& SampleToken);
    int getSampleIdGivenLidarTop(const std::string& LidarTop);
    const vec getAnnotationColor(const std::string& annotation);
    void showResults();
    void distortionCorrection(float*& data, unsigned int& length, float& yaw, float& pitch, float& roll);
    SphericalCoordinates convertToSpherical(const vec &point);

    std::vector<Annotation> getAnnotations();
    std::vector<Sample>  getSamples();
    std::vector<Scene> getScenes();
    void getMapSamples(std::map<std::string, int>& m_mapSamplesWithId);
};

#endif //LAB0_LYFTDATASETHANDLER_H