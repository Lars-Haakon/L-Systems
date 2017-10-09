#include <GL/glew.h>
#include <fstream>

#include "shader.h"

Shader::Shader()
{
	m_program = glCreateProgram();
}

Shader::~Shader()
{
	glDeleteProgram(m_program);
}

void Shader::CompileShader()
{
	glLinkProgram(m_program);
}

void Shader::Bind()
{
	glUseProgram(m_program);
}

void Shader::AddVertexShader(std::string fileName)
{
	AddProgram(LoadShaderSource(fileName), GL_VERTEX_SHADER);
}

void Shader::AddGeometryShader(std::string fileName)
{
	AddProgram(LoadShaderSource(fileName), GL_GEOMETRY_SHADER);
}

void Shader::AddFragmentShader(std::string fileName)
{
	AddProgram(LoadShaderSource(fileName), GL_FRAGMENT_SHADER);
}

int Shader::AddUniform(const char* uniform)
{
	return glGetUniformLocation(m_program, uniform);
}

void Shader::SetUniformMat4(int uniformLocation, const float* matrix)
{
	glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, matrix);
}

void Shader::SetUniformVec3(int uniformLocation, float x, float y, float z)
{
    glUniform3f(uniformLocation, x, y, z);
}

void Shader::AddProgram(std::string shaderSource, int type)
{
	int shader = glCreateShader(type);

	if (shader == 0)
	{
		printf("%s %d\n", "Error creating shader type", type);
	}

	const char* source = shaderSource.c_str();
	int sourceLength = shaderSource.length();
	glShaderSource(shader, 1, &source, &sourceLength);

	glCompileShader(shader);

	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		char infoLog[1024];
		glGetShaderInfoLog(shader, 1024, NULL, infoLog);
		printf("Error compiling shader type %d %s\n", shader, infoLog);
	}

	glAttachShader(m_program, shader);
}

std::string Shader::LoadShaderSource(std::string fileName)
{
	std::ifstream file(("./shaders/" + fileName).c_str());
	std::string shaderSource((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

	return shaderSource;
}
