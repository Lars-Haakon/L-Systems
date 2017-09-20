#ifndef SHADER_H
#define SHADER_H

#include <glm/glm.hpp>
#include <string>

class Shader
{
public:
	Shader();
	~Shader();

	void CompileShader();
	void Bind();

	void AddVertexShader(std::string filename);
    void AddGeometryShader(std::string filename);
	void AddFragmentShader(std::string filename);

	int AddUniform(const char* uniform);
	void SetUniformMat4(int uniformLocation, const float* matrix);
private:
	void AddProgram(std::string shaderSource, int type);
	std::string LoadShaderSource(std::string fileName);

	int m_program;
};

#endif
