#include "Renderer.h"
#include "DataStructures.h"

void GLClearError() {
    while (glGetError() != GL_NO_ERROR);
}

bool GLLogCall(const char* function, const char* file, int line) {
    while (GLenum error = glGetError())
    {
        std::cout << "[OpenGL Error] (" << error << "): " << function << " " << file << ":" << line << std::endl;
        return false;
    }
    return true;
}

/// without indices, without instances
void Renderer::Draw(const VertexArray &va, const Shader &shader, const unsigned int count, GLenum mode) const {
    /// bind objects
    shader.Bind();
    va.Bind();

    /// draw data
    GLCall(glDrawArrays(mode, 0, count));
}

/// with indices, without instances
void Renderer::Draw(const VertexArray &va, const IndexBuffer &ib, const Shader &shader, GLenum mode) const {
    /// bind objects
    shader.Bind();
    va.Bind();
    ib.Bind();

    /// draw data
    GLCall(glDrawElements(mode, ib.GetCount(), GL_UNSIGNED_INT, nullptr));
}

/// without indices, with instances
/// count => Specifies the number of indices to be rendered.
/// instances => Specifies the number of instances of the specified range of indices to be rendered.
void Renderer::DrawInstances(const VertexArray &va, const Shader &shader, const unsigned int count, const unsigned int instances, GLenum mode) const {
    /// bind objects
    shader.Bind();
    va.Bind();

    /// draw data
    GLCall(glDrawArraysInstanced(mode, 0, count, instances)); // draw instances triangles of count vertices each)
}

void Renderer::Clear() const {
//    GLCall(glClearColor(0.2f, 0.3f, 0.3f, 0.5f));

    GLCall(glClearColor(0.0f, 0.0f, 0.0f, 0.0f));

    GLCall(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
}

/// create indices in order to draw lines of boxes
void Renderer::CreateBoxIndices(unsigned int* indices, const int& NumBoxes) {
    int pointIndex{0}, lineIndex{0}, lines{12};
    Box3DPoints boxPoints{};
    for (unsigned int i=0; i<NumBoxes; i++) {
        /// define the four corners which face forward
        boxPoints.topRightF = pointIndex; boxPoints.topLeftF = pointIndex+1;
        boxPoints.bottomLeftF = pointIndex+2; boxPoints.bottomRightF = pointIndex+3;

        /// define the four corners which face backwards
        boxPoints.topRightB = pointIndex+4; boxPoints.topLeftB = pointIndex+5;
        boxPoints.bottomLeftB = pointIndex+6; boxPoints.bottomRightB = pointIndex+7;

        /// create 3 lines
        indices[lineIndex++] = boxPoints.topRightF; indices[lineIndex++] = boxPoints.topLeftF;
        indices[lineIndex++] = boxPoints.topRightF; indices[lineIndex++] = boxPoints.bottomRightF;
        indices[lineIndex++] = boxPoints.topRightF; indices[lineIndex++] = boxPoints.topRightB;

        /// create 2 lines
        indices[lineIndex++] = boxPoints.topLeftF; indices[lineIndex++] = boxPoints.bottomLeftF;
        indices[lineIndex++] = boxPoints.topLeftF; indices[lineIndex++] = boxPoints.topLeftB;

        /// create 2 lines
        indices[lineIndex++] = boxPoints.bottomLeftF; indices[lineIndex++] = boxPoints.bottomRightF;
        indices[lineIndex++] = boxPoints.bottomLeftF; indices[lineIndex++] = boxPoints.bottomLeftB;

        /// create 1 line
        indices[lineIndex++] = boxPoints.bottomRightF; indices[lineIndex++] = boxPoints.bottomRightB;

        /// create 2 lines
        indices[lineIndex++] = boxPoints.topRightB; indices[lineIndex++] = boxPoints.topLeftB;
        indices[lineIndex++] = boxPoints.topRightB; indices[lineIndex++] = boxPoints.bottomRightB;

        /// create 1 line
        indices[lineIndex++] = boxPoints.topLeftB; indices[lineIndex++] = boxPoints.bottomLeftB;

        /// create 1 line
        indices[lineIndex++] = boxPoints.bottomLeftB; indices[lineIndex++] = boxPoints.bottomRightB;

        pointIndex += 8;
    }
}

float* Renderer::setData(const pointCloudBoost &cloud, const vec &col, const unsigned int& length) {
    float* data = new float[length];
    int index{0};
    for (unsigned int i=0; i<cloud->width*cloud->height; i++) {
        *(data+index) = cloud->points[i].x; *(data+index+1) = cloud->points[i].y; *(data+index+2) = cloud->points[i].z;
        *(data+index+3) = col.x; *(data+index+4) = col.y; *(data+index+5) = col.z;
        index += 6;
    }
    return data;
}