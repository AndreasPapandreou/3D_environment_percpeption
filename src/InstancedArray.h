#ifndef LAB0_INSTANCEDARRAY_H
#define LAB0_INSTANCEDARRAY_H

#include <glm/matrix.hpp>

class InstancedArray {
private:
    unsigned int m_RenderedID;
    unsigned int m_Count;
public:
    InstancedArray(const glm::vec2* data, unsigned int count);
    ~InstancedArray();

    void Bind() const;
    void Unbind() const;

    inline unsigned int GetCount() const { return m_Count; }
};

#endif //LAB0_INSTANCEDARRAY_H