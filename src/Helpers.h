#define GLM_ENABLE_EXPERIMENTAL
#ifndef LAB0_HELPERS_H
#define LAB0_HELPERS_H

#include "DataStructures.h"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>

/// store all types of categories and add an index per category
struct types
{
    std::string name;
    int index;
    types(std::string name, int index): name(std::move(name)), index(index) {};
};

class Helpers {
public:
    static double dotProduct(const vec& v1, const vec& v2);
    bool pointInBox(const vec& p_check, const VecArray& corners);
    static vec getCenter(const VecArray& data);
    static int existInVector(std::vector<types>& data, const std::string& value);
    std::string getCurrentDir();
//    SphericalCoordinates convertToSperical(vec point);

    void getBoxes(VecArray& borders, std::string& pathToVehicles);

    void projectToIm(const VecArray& lidar, VecArray& camera, calibMat& calib, vec& egoLocation, float yaw=0.0f, float pitch=0.0f, float roll=0.0f);
    void rotate_data(VecArray& points, float angle_radian, glm::vec3 myRotationAxis);
    void rotate_data(vec &p, float angle_radian, glm::vec3 myRotationAxis);
};

#endif //LAB0_HELPERS_H