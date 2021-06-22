#include "InstancedArray.h"
#include "Renderer.h"

InstancedArray::InstancedArray(const glm::vec2* data, unsigned int count)
        : m_Count(count) {
    ASSERT(sizeof(unsigned int) == sizeof(GLuint));

    GLCall(glGenBuffers(1, &m_RenderedID));
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RenderedID));
    GLCall(glBufferData(GL_ARRAY_BUFFER, count * sizeof(unsigned int), data, GL_STATIC_DRAW));
}

InstancedArray::~InstancedArray() {
    GLCall(glDeleteBuffers(1, &m_RenderedID));
}

void InstancedArray::Bind() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RenderedID));
}

void InstancedArray::Unbind() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}