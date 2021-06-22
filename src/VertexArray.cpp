#include "VertexArray.h"
#include "Renderer.h"
#include "VertexBufferLayout.h"
#include "InstancedArray.h"
#include "InstancedArrayLayout.h"

VertexArray::VertexArray() {
    GLCall(glGenVertexArrays(1, &m_RendererID));
}

VertexArray::~VertexArray() {
    GLCall(glDeleteVertexArrays(1, &m_RendererID));
}

//void VertexArray::AddVertexBuffer(const VertexBuffer &vb, const VertexBufferLayout &layout) {
//    /// bind vertex array
//    Bind();
//
//    /// bind vertex buffer
//    vb.Bind();
//
//    /// set up the layout
//    const auto& elements = layout.GetElements();
//    unsigned int offset = 0;
//    for (unsigned int i = 0; i<elements.size(); i++)
//    {
//        const auto& element = elements[i];
//
//        /// enables a generic vertex attribute array
//        GLCall(glEnableVertexAttribArray(i));
//
//        /// defines an array of generic vertex attribute data
//        GLCall(glVertexAttribPointer(i, element.count, element.type, element.normalized,
//                                     layout.GetStride(), (const void*)offset));
//
//        offset += element.count * VertexBufferElement::GetSizeOfType(element.type);
//    }
//}

void VertexArray::AddVertexBuffer(const VertexBuffer &vb, const VertexBufferLayout &layout, const unsigned int CurrentLayout) {
    /// bind vertex array
    Bind();

    /// bind vertex buffer
    vb.Bind();

    /// set up the layout
    const auto& elements = layout.GetElements();
    unsigned int offset = 0;
    for (unsigned int i = CurrentLayout; i<elements.size()+CurrentLayout; i++)
    {
        const auto& element = elements[i-CurrentLayout];

        /// enables a generic vertex attribute array
        GLCall(glEnableVertexAttribArray(i));

        /// defines an array of generic vertex attribute data
        GLCall(glVertexAttribPointer(i, element.count, element.type, element.normalized,
                                     layout.GetStride(), (const void*)offset));

        offset += element.count * VertexBufferElement::GetSizeOfType(element.type);
    }
}

void VertexArray::AddInstancedArray(const InstancedArray &ia, const InstancedArrayLayout &layout, const unsigned int CurrentLayout) {
    /// bind vertex array
    Bind();

    /// bind instanced array
    ia.Bind();

    /// set up the layout
    const auto& elements = layout.GetElements();
    unsigned int offset = 0;
    for (unsigned int i = CurrentLayout; i<elements.size()+CurrentLayout; i++)
    {
        const auto& element = elements[i-CurrentLayout];

        /// enables a generic vertex attribute array
        GLCall(glEnableVertexAttribArray(i));

        /// defines an array of generic vertex attribute data
        GLCall(glVertexAttribPointer(i, element.count, element.type, element.normalized,
                                     layout.GetStride(), (const void*)offset));

        offset += element.count * VertexBufferElement::GetSizeOfType(element.type);

        /// By setting this attribute to 1 we're telling OpenGL that we want to update the content of the vertex
        /// attribute when we start to render a new instance. By setting it to 2 we'd update the content every
        /// 2 instances and so on ...
        GLCall(glVertexAttribDivisor(i, 1));
    }
}

void VertexArray::Bind() const {
    GLCall(glBindVertexArray(m_RendererID));
}

void VertexArray::Unbind() const {
    GLCall(glBindVertexArray(0));
}