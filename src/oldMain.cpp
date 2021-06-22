//
//#include <glad/glad.h>
//#include <GLFW/glfw3.h>
//#include <glm/matrix.hpp>
//#include <iostream>
//#include <alloca.h>
//
//#include "Renderer.h"
//#include "VertexBuffer.h"
//#include "IndexBuffer.h"
//#include "VertexArray.h"
//#include "Shader.h"
//#include "VertexBufferLayout.h"
//#include "InstancedArray.h"
//#include "Camera.h"
//
//void framebuffer_size_callback(GLFWwindow* window, int width, int height);
//void processInput(GLFWwindow *window);
//
///// settings
//const unsigned int SCR_WIDTH = 800;
//const unsigned int SCR_HEIGHT = 600;
//
//int main() {
//    float vertices[] = {
//            // positions     // colors
//            -0.05f,  0.05f,  1.0f, 0.5f, 0.0f,
//            0.05f, -0.05f,  0.0f, 1.0f, 0.0f,
//            -0.05f, -0.05f,  0.0f, 0.1f, 1.0f,
//
//            -0.05f,  0.05f,  1.0f, 0.0f, 0.0f,
//            0.05f, -0.05f,  0.0f, 1.0f, 0.0f,
//            0.05f,  0.05f,  0.0f, 1.0f, 1.0f
//    };
//
//    /// define some indices
//    unsigned int indices[] = {
//            0, 1, 2,
//            3, 4, 5
//    };
//
//    // generate a list of 100 quad locations/translation-vectors
//    // ---------------------------------------------------------
//    glm::vec2 translations[100];
//    int index = 0;
//    float offset = 0.1f;
//    for (int y = -10; y < 10; y += 2)
//    {
//        for (int x = -10; x < 10; x += 2)
//        {
//            glm::vec2 translation;
//            translation.x = (float)x / 10.0f + offset;
//            translation.y = (float)y / 10.0f + offset;
//            translations[index++] = translation;
//        }
//    }
//
//    VertexArray va;
//
//    VertexBuffer vb(vertices, sizeof(vertices));
//
//    VertexBufferLayout layoutVertex;
//    layoutVertex.Push(2); /// the first attribute (position) has count floats
//    layoutVertex.Push(3);
//    va.AddVertexBuffer(vb, layoutVertex);
//
//    InstancedArray ia(translations, 100 * sizeof(float));
//    InstancedArrayLayout layoutArray;
//    layoutArray.Push(2); /// the first attribute (position) has count floats
//    va.AddInstancedArray(ia, layoutArray, layoutVertex.GetSize());
//
//    IndexBuffer ib(indices, 6);
//
//    Shader shader("res/shaders/Basic.shader");
//
//    Renderer renderer;
//
//    /// render loop
//    while (!glfwWindowShouldClose(window))
//    {
//        /// input
//        processInput(window);
//
//        renderer.Clear();
//        shader.Bind();
//        renderer.Draw(va, ib, shader);
////        renderer.DrawInstances(va, shader, 6, sizeof(translations)/(2*sizeof(float)));
//
////        camera->update();
//
//        glfwSwapBuffers(window);
//        glfwPollEvents();
//    }
//    glfwTerminate();
//    return 0;
//}