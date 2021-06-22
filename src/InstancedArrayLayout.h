#include <vector>
#include <cassert>
#include <glad/glad.h>

#ifndef LAB0_INSTANCEDARRAYLAYOUT_H
#define LAB0_INSTANCEDARRAYLAYOUT_H

struct InstancedArrayElement
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
//        ASSERT(false); //TODO
        return 0;
    }
};

/// *** NOTE ***
/// The vertex buffer layout accepts only FLOATS

class InstancedArrayLayout {
private:
    std::vector<InstancedArrayElement> m_Elements;
    unsigned int m_Stride;

public:
    InstancedArrayLayout()
            : m_Stride(0) {}

    /// type => 0 for VertexBuffer and 1 for InstancedArray
    void Push(unsigned int count)
    {
        m_Elements.push_back({ GL_FLOAT, count, GL_FALSE });
        m_Stride += count * InstancedArrayElement::GetSizeOfType(GL_FLOAT);
    }

    inline const std::vector<InstancedArrayElement> GetElements() const& { return m_Elements; }
    inline unsigned int GetStride() const { return m_Stride; }
};


#endif //LAB0_INSTANCEDARRAYLAYOUT_H
