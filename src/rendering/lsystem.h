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
    LSystem(std::string axiom, float s0, float w0,
                                float r1, float r2,
                                float a1, float a2,
                                float t1, float t2,
                                float q, float e,
                                float min, Transform transform = Transform());
    ~LSystem();

    void AddProduction(char predecessor, std::string successor);
    void Generate(int n);

    void Draw();
private:
    std::string GetSuccessor(char predecessor);

    std::string m_axiom;
    float m_s0;
    float m_w0;
    float m_r1;
    float m_r2;
    float m_a1;
    float m_a2;
    float m_t1;
    float m_t2;
    float m_q;
    float m_e;
    float m_min;
    std::vector<Production> m_productions;

    unsigned int m_vbo;
    unsigned int m_ibo;
    int m_size;
};

#endif
