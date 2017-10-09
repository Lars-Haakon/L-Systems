#include <GL/glew.h>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "lsystem.h"
#include "../cuda/cudainfo.cuh"

LSystem::LSystem(std::string axiom, float s0, float w0,
                            float r1, float r2,
                            float a1, float a2,
                            float t1, float t2,
                            float q, float e,
                            float min, Transform transform)
: SceneObject(transform)
{
    m_axiom = axiom;
    m_s0 = s0;
    m_w0 = w0;
    m_r1 = r1;
    m_r2 = r2;
    m_a1 = a1;
    m_a2 = a2;
    m_t1 = t1;
    m_t2 = t2;
    m_q = q;
    m_e = e;
    m_min = min;

	glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ibo);
}

LSystem::~LSystem()
{
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &m_ibo);
}

void LSystem::AddProduction(char predecessor, std::string successor)
{
    Production p;
    p.predecessor = predecessor;
    p.successor = successor;
    m_productions.push_back(p);
}

void LSystem::Generate(int n)
{
    std::vector<int> parameterIndices;
    std::vector<float> parameters;

    std::string generatedString = m_axiom;
    parameters.push_back(m_s0);
    parameters.push_back(m_w0);
    parameterIndices.push_back(0);

    for(int i = 0; i < n; i++)
    {
        std::string iterationString = "";

        for(std::string::iterator node = generatedString.begin(); node != generatedString.end(); node++)
        {
            iterationString += GetSuccessor(*node);
        }

        generatedString = iterationString;
    }

    printf("Module length: %d\n", (int)generatedString.length());

    int lookUpTableSize = 128*16*sizeof(float); // 128 matrices, 16 floats each matrix
    float* lookUpTable = (float*)malloc(lookUpTableSize);
    const float* identity =  glm::value_ptr(glm::mat4(1.0f));
    for(int m = 0; m < 128; m++)
    {
        for(int i = 0; i < 16; i++)
            lookUpTable[m*16+i] = identity[i];
    }
    /*
    const float* F = glm::value_ptr(glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -m_distance)));
    for(int i = 0; i < 16; i++)
        lookUpTable['F'*16+i] = F[i];

    const float* plus = glm::value_ptr(glm::rotate(glm::mat4(1.0f), glm::radians(m_angle), glm::vec3(0,1,0)));
    for(int i = 0; i < 16; i++)
        lookUpTable['+'*16+i] = plus[i];

    const float* minus = glm::value_ptr(glm::rotate(glm::mat4(1.0f), glm::radians(-m_angle), glm::vec3(0,1,0)));
    for(int i = 0; i < 16; i++)
        lookUpTable['-'*16+i] = minus[i];

    const float* ampersand = glm::value_ptr(glm::rotate(glm::mat4(1.0f), glm::radians(m_angle), glm::vec3(1,0,0)));
    for(int i = 0; i < 16; i++)
        lookUpTable['&'*16+i] = ampersand[i];

    const float* caret = glm::value_ptr(glm::rotate(glm::mat4(1.0f), glm::radians(-m_angle), glm::vec3(1,0,0)));
    for(int i = 0; i < 16; i++)
        lookUpTable['^'*16+i] = caret[i];

    const float* backslash = glm::value_ptr(glm::rotate(glm::mat4(1.0f), glm::radians(m_angle), glm::vec3(0,0,1)));
    for(int i = 0; i < 16; i++)
        lookUpTable['\\'*16+i] = backslash[i];

    const float* slash = glm::value_ptr(glm::rotate(glm::mat4(1.0f), glm::radians(-m_angle), glm::vec3(0,0,1)));
    for(int i = 0; i < 16; i++)
        lookUpTable['/'*16+i] = slash[i];

    const float* vertical = glm::value_ptr(glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0,1,0)));
    for(int i = 0; i < 16; i++)
        lookUpTable['|'*16+i] = vertical[i];
    */

    int num_lines = FillData(lookUpTable, lookUpTableSize, generatedString.c_str(), generatedString.length());
    int num_vertices = num_lines+1;
    m_size = 2*num_lines;

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices * sizeof(float), 0, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_size * sizeof(int), 0, GL_STATIC_DRAW);

    FillBuffers(m_vbo, m_ibo, generatedString.length());

    free(lookUpTable);
}

void LSystem::Draw()
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    glDrawElements(GL_LINES, m_size, GL_UNSIGNED_INT, 0);
}

std::string LSystem::GetSuccessor(char predecessor)
{
    for(int i = 0; i < m_productions.size(); i++)
    {
        if(m_productions[i].predecessor == predecessor)
            return m_productions[i].successor;
    }

    return std::string(&predecessor, 1);
}
