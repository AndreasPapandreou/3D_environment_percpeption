#ifndef LAB0_VERTEXBUFFER_H
#define LAB0_VERTEXBUFFER_H

class VertexBuffer {
private:
    unsigned int m_RenderedID;
public:
    VertexBuffer(const void* data, unsigned int size);
    ~VertexBuffer();

    void Bind() const;
    void Unbind() const;
};

#endif //LAB0_VERTEXBUFFER_H