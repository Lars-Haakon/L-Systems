#ifndef LSYSTEM_H
#define LSYSTEM_H
#include <vector>
#include <string>

#include "../core/sceneobject.h"

struct Production
{
    char predecessor;
    std::string successor;
};

class LSystem : public SceneObject
{
public:
    LSystem(std::string axiom, float distance, float angle, Transform transform = Transform());
    ~LSystem();

    void AddProduction(char predecessor, std::string successor);
    void Generate(int n);

    void Draw();
private:
    std::string GetSuccessor(char predecessor);

    std::string m_axiom;
    float m_distance;
    float m_angle;
    std::vector<Production> m_productions;

    unsigned int m_vbo;
    unsigned int m_ibo;
    int m_size;
};

#endif
