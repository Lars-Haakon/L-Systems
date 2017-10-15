#include "texture.h"

#include <GL/glew.h>
#include <fstream>

Texture::Texture(std::string fileName)
{
    glGenTextures(1, &m_id);
	Bind();

    std::ifstream file(("./textures/" + fileName).c_str());
	std::string shaderSource((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    const char* header = shaderSource.c_str();

	int dataPos = *(int*)&(header[0xA]);
	int imageSize = *(int*)&(header[0x22]);
	int width = *(int*)&(header[0x12]);
	int height = *(int*)&(header[0x16]);

    printf("ImageSize: %d\n", imageSize);
    printf("Width: %d\n", width);
    printf("Height: %d\n", height);

	if (imageSize == 0)
		imageSize = width*height * 3; // 3 : one byte for each Red, Green and Blue component
	if (dataPos == 0)
		dataPos = 54;

	// allocate buffer
	const char* data = &header[54];

	//Setup wrap mode
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	//Setup texture scaling filtering
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);

    glGenerateMipmap(GL_TEXTURE_2D);
    GLfloat maxAnisotropy;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAnisotropy);

    file.close();
}

Texture::~Texture()
{
    glDeleteTextures(1, &m_id);
}

void Texture::Bind()
{
    glBindTexture(GL_TEXTURE_2D, m_id);
}
