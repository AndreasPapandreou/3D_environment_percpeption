#define GLM_ENABLE_EXPERIMENTAL
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <cmath>

#include "Camera.h"
#include <glm/gtx/transform.hpp>

/// Constructor with vectors
Camera::Camera(glm::vec3 position, glm::vec3 worldUp, float yaw, float pitch)
{
    m_Position = position;
    m_WorldUp = worldUp;
    m_Yaw = yaw;
    m_Pitch = pitch;
    shiftPressed = false;
    updateCameraVectors();
}

void Camera::setEgoPos(glm::vec3 position) {
    m_Position = position;
}

/// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
glm::mat4 Camera::GetViewMatrix()
{
    return glm::lookAt(m_Position, m_Position + m_Front, m_Up);
}

/// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
void Camera::processInput(GLFWwindow *window)
{

    if (shiftPressed) {
        m_MovementSpeed /= 10.0f;
        shiftPressed = false;
    }

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        std::cout << glm::to_string(m_Position) << std::endl;

    if (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        shiftPressed = true;
        m_MovementSpeed *= 10.0f;
    }

    float cameraSpeed = m_MovementSpeed * deltaTime;
    /// move forward
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        m_Position += cameraSpeed * m_Up;
    /// move back
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        m_Position -= cameraSpeed * m_Up;
    /// move left
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        m_Position -= cameraSpeed * m_Right;
    /// move right
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        m_Position += cameraSpeed * m_Right;
    /// zoom in
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        m_Position += cameraSpeed * m_Front;
    /// zoom out
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
        m_Position -= cameraSpeed * m_Front;

    float degrees = 1.0f;
    float radians = degrees*M_PI/180;
    /// rotate left
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        glm::vec3 myRotationAxis(0.0, 0.0, 1.0);
        glm::mat4 rotation = glm::rotate(radians, myRotationAxis);
        rotateCamera(rotation);
    }
    /// rotate right
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        glm::vec3 myRotationAxis(0.0, 0.0, 1.0);
        glm::mat4 rotation = glm::rotate(-radians, myRotationAxis);
        rotateCamera(rotation);
    }
    /// rotate up
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        glm::vec3 myRotationAxis(1.0, 0.0, 0.0);
        glm::mat4 rotation = glm::rotate(radians, myRotationAxis);
        rotateCamera(rotation);
    }
    /// rotate down
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        glm::vec3 myRotationAxis(1.0, 0.0, 0.0);
        glm::mat4 rotation = glm::rotate(-radians, myRotationAxis);
        rotateCamera(rotation);
    }
}

void Camera::rotateCamera(glm::mat4 rotation) {
    m_Up = glm::mat3(rotation)* m_Up;
    m_Right = glm::mat3(rotation)* m_Right;
    m_Front = glm::mat3(rotation)* m_Front;
}

/// Calculates the front vector from the Camera's (updated) Euler Angles
void Camera::updateCameraVectors()
{
    /// Calculate the new Front vector
//    m_Yaw *= M_PI / 180.0f; /// convert degree to radian
//    m_Pitch *= M_PI / 180.0f; /// convert degree to radian

    glm::vec3 front;
    front.x = cos(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
    front.y = sin(glm::radians(m_Pitch));
    front.z = sin(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
    m_Front = glm::normalize(front);

    /// Also re-calculate the Right and Up vector
    m_Right = glm::normalize(glm::cross(m_Front, m_WorldUp));  /// Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    m_Up    = glm::normalize(glm::cross(m_Right, m_Front));
}

/// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
//void Camera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
//{
//    xoffset *= m_MouseSensitivity;
//    yoffset *= m_MouseSensitivity;
//
//    m_Yaw   += xoffset;
//    m_Pitch += yoffset;
//
//    /// Make sure that when pitch is out of bounds, screen doesn't get flipped
//    if (constrainPitch)
//    {
//        if (m_Pitch > 89.0f)
//            m_Pitch = 89.0f;
//        if (m_Pitch < -89.0f)
//            m_Pitch = -89.0f;
//    }
//
//    /// Update Front, Right and Up Vectors using the updated Euler angles
//    updateCameraVectors();
//}

/// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
//void Camera::ProcessMouseScroll(float yoffset)
//{
//    if (m_Fov >= 1.0f && m_Fov <= 45.0f)
//        m_Fov -= yoffset;
//    if (m_Fov <= 1.0f)
//        m_Fov = 1.0f;
//    if (m_Fov >= 45.0f)
//        m_Fov = 45.0f;
//}