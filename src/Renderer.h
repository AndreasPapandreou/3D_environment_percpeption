#ifndef LAB0_RENDERER_H
#define LAB0_RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"
#include "DataStructures.h"

#define ASSERT(x) if (!(x)) __builtin_trap();

/// #x --> converts the function to string
#define GLCall(x) GLClearError(); x; ASSERT(GLLogCall(#x, __FILE__, __LINE__))

void GLClearError();
bool GLLogCall(const char* function, const char* file, int line);

class Renderer {
public:
    void Clear() const;
    void Draw(const VertexArray& va, const IndexBuffer& ib, const Shader& shader, GLenum mode) const;
    void Draw(const VertexArray& va, const Shader& shader, unsigned int count, GLenum mode) const;
    void DrawInstances(const VertexArray &va, const Shader &shader, unsigned int count, unsigned int instances, GLenum mode) const;
    void CreateBoxIndices(unsigned int* indices, const int& NumBoxes);
    float* setData(const pointCloudBoost& cloud, const vec& col, const unsigned int& size);

};

#endif //LAB0_RENDERER_H