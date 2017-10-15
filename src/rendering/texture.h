#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>

class Texture
{
public:
    Texture(std::string fileName);
    ~Texture();

    void Bind();

private:
    unsigned int m_id;
};

#endif
