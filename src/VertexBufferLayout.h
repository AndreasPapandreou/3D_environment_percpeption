#include <vector>
#include <cassert>
//#include "Renderer.h"

#ifndef LAB0_VERTEXBUFFERLAYOUT_H
#define LAB0_VERTEXBUFFERLAYOUT_H

struct VertexBufferElement
{
    unsigned int type;
    unsigned int count;
    unsigned char normalized;

    static unsigned int GetSizeOfType(unsigned int type)
    {
        switch (type)
        {
            case GL_FLOAT:  return 4;
        }
//        ASSERT(false);
        return 0;
    }
};

/// *** NOTE ***
/// The vertex buffer layout accepts only FLOATS

class VertexBufferLayout {
private:
    std::vector<VertexBufferElement> m_Elements;
    unsigned int m_Stride;

public:
    VertexBufferLayout()
        : m_Stride(0) {}

    /// type => 0 for VertexBuffer and 1 for InstancedArray
    void Push(unsigned int count)
    {
        m_Elements.push_back({ GL_FLOAT, count, GL_FALSE });
        m_Stride += count * VertexBufferElement::GetSizeOfType(GL_FLOAT);
    }

    inline const std::vector<VertexBufferElement> GetElements() const& { return m_Elements; }
    inline unsigned int GetStride() const { return m_Stride; }
    inline unsigned int GetSize() const { return m_Elements.size(); }
};

#endif //LAB0_VERTEXBUFFERLAYOUT_H