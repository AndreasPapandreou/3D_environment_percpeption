#ifndef LAB0_VERTEXARRAY_H
#define LAB0_VERTEXARRAY_H

#include "VertexBuffer.h"
#include "InstancedArray.h"
#include "InstancedArrayLayout.h"

class VertexBufferLayout;

class VertexArray {
private:
    unsigned int m_RendererID;
public:
    VertexArray();
    ~VertexArray();
    void AddVertexBuffer(const VertexBuffer& vb, const VertexBufferLayout& layout, unsigned int CurrentLayout=0);
    void AddInstancedArray(const InstancedArray& ia, const InstancedArrayLayout& layout, unsigned int CurrentLayout);

    void Bind() const;
    void Unbind() const;
};

#endif //LAB0_VERTEXARRAY_H