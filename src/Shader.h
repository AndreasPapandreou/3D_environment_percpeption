#ifndef LAB0_SHADER_H
#define LAB0_SHADER_H

#include <string>
#include <unordered_map>
#include <glm/glm.hpp>

struct ShaderProgramSource
{
    std::string VertexSource;
    std::string GeometrySource;
    std::string FragmentSource;
};

class Shader {

private:
    std::string m_FilePath;
    unsigned int m_RendererID;

    /// caching for uniforms
    std::unordered_map<std::string, int> m_UniformLocationCache;

public:
    Shader(const std::string& filepath, bool usingGeometry = false);
    ~Shader();

    void Bind() const;
    void Unbind() const;

    /// Set uniforms
    void SetUniform4f(const std::string& name, float v0, float v1, float v2, float v3);
    void SetMat4(const std::string &name, const glm::mat4 &mat);
private:
    ShaderProgramSource ParseShader(const std::string& filepath);
    unsigned int CompileShader(unsigned int type, const std::string& source);
    unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);
    unsigned int CreateShader2(const std::string& vertexShader, const std::string& geometrySource, const std::string& fragmentShader);
    int GetUniformLocation(const std::string& name);
};

#endif //LAB0_SHADER_H