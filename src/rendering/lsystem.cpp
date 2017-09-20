#include <GL/glew.h>
#include <stdio.h>

#include "lsystem.h"

LSystem::LSystem(std::string axiom, float distance, float angle, Transform transform)
: SceneObject(transform)
{
    m_axiom = axiom;
    m_distance = distance;
    m_angle = angle;

	glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ibo);
}

LSystem::~LSystem()
{
    glDeleteBuffers(1, &m_vbo);
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
    std::string generatedString = m_axiom;

    for(int i = 0; i < n; i++)
    {
        std::string iterationString = "";

        for(std::string::iterator node = generatedString.begin(); node != generatedString.end(); node++)
        {
            iterationString += GetSuccessor(*node);
        }

        generatedString = iterationString;
    }

    std::vector<float> vertices;
    std::vector<int> indices;

    Transform turtle(glm::vec3(0, 0, 0), glm::normalize(glm::quat(1, 1, 0, 0)));

    bool drawLine = false;
    std::vector<int> indexStack(1, 0);
    int nextIndex = 1;

    std::vector<Transform> turtleStack;
    for(std::string::iterator node = generatedString.begin(); node != generatedString.end(); node++)
    {
        switch(*node)
        {
            case 'F':
            {
                drawLine = true;

                glm::vec3 pos = turtle.GetPosition();
                vertices.push_back(pos[0]);
                vertices.push_back(pos[1]);
                vertices.push_back(pos[2]);

                turtle.SetPosition(turtle.GetPosition() + turtle.Forward() * m_distance);

                indices.push_back(indexStack.back());
                indices.push_back(nextIndex++);
                indexStack.pop_back();
                indexStack.push_back(indices.back());

                break;
            }
            case 'f':
            {
                if(drawLine)
                {
                    glm::vec3 pos = turtle.GetPosition();
                    vertices.push_back(pos[0]);
                    vertices.push_back(pos[1]);
                    vertices.push_back(pos[2]);

                    indexStack.pop_back();
                    indexStack.push_back(nextIndex++);
                }
                drawLine = false;

                turtle.SetPosition(turtle.GetPosition() + turtle.Forward() * m_distance);
                break;
            }
            case '+':
            {
                turtle.SetRotation(glm::angleAxis(glm::radians(m_angle), turtle.Up()) * turtle.GetRotation());
                break;
            }
            case '-':
            {
                turtle.SetRotation(glm::angleAxis(glm::radians(-m_angle), turtle.Up()) * turtle.GetRotation());
                break;
            }
            case '&':
            {
                turtle.SetRotation(glm::angleAxis(glm::radians(m_angle), turtle.Right()) * turtle.GetRotation());
                break;
            }
            case '^':
            {
                turtle.SetRotation(glm::angleAxis(glm::radians(-m_angle), turtle.Right()) * turtle.GetRotation());
                break;
            }
            case '\\':
            {
                turtle.SetRotation(glm::angleAxis(glm::radians(m_angle), turtle.Forward()) * turtle.GetRotation());
                break;
            }
            case '/':
            {
                turtle.SetRotation(glm::angleAxis(glm::radians(-m_angle), turtle.Forward()) * turtle.GetRotation());
                break;
            }
            case '|':
            {
                turtle.SetRotation(glm::angleAxis(glm::radians(180.0f), turtle.Up()) * turtle.GetRotation());
                break;
            }
            case '[':
            {
                turtleStack.push_back(turtle);

                indexStack.push_back(indexStack.back());
                break;
            }
            case ']':
            {
                glm::vec3 pos = turtle.GetPosition();
                vertices.push_back(pos[0]);
                vertices.push_back(pos[1]);
                vertices.push_back(pos[2]);

                turtle = turtleStack.back();
                turtleStack.pop_back();

                indexStack.pop_back();
                nextIndex++;
                break;
            }
        }
    }

    glm::vec3 pos = turtle.GetPosition();
    vertices.push_back(pos[0]);
    vertices.push_back(pos[1]);
    vertices.push_back(pos[2]);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(int), &indices[0], GL_STATIC_DRAW);

	m_size = indices.size();
}

void LSystem::Draw()
{
    glBindVertexArray(m_ibo);
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
