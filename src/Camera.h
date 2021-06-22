#ifndef LAB0_CAMERA_H
#define LAB0_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glad/glad.h>
#include <vector>
#include "libs.h"

/// Default camera values
const float YAW         = -90.0f;
const float PITCH       =  0.0f;
const float SPEED       =  4.5f;
const float FOV         =  45.0f;

/// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera
{
public:
    /// Camera Attributes
    glm::vec3 m_Position;
    glm::vec3 m_WorldUp;
    glm::vec3 m_Front;
    glm::vec3 m_Right;
    glm::vec3 m_Up;

    /// Euler Angles
    float m_Yaw;
    float m_Pitch;

    /// Timing
    float deltaTime = 0.0f;	// Time between current frame and last frame
    float lastFrame = 0.0f; // Time of last frame

    /// Camera options
    float m_MovementSpeed = SPEED;
    float m_Fov = FOV;

    bool shiftPressed;

    /// Constructor with vectors
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
//            glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3 worldUp = glm::vec3(1.0f, 0.0f, 0.0f),
            float yaw = YAW, float pitch = PITCH);

    /// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix();
    /// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void processInput(GLFWwindow *window);
    void setEgoPos(glm::vec3 position);
    /// Calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors();
    void rotateCamera(glm::mat4 rotation);
};
#endif