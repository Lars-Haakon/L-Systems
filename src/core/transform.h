#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

class Transform
{
public:
	Transform(	glm::vec3 position = glm::vec3(0, 0, 0), 
			glm::quat rotation = glm::quat(1, 0, 0, 0),
			glm::vec3 scale = glm::vec3(1, 1, 1));
	~Transform();

	glm::mat4 GetModelMatrix();
	glm::vec3 GetPosition();
	glm::quat GetRotation();
	glm::vec3 GetScale();

	void SetPosition(glm::vec3 position);
	void SetRotation(glm::quat rotation);
	void SetScale(glm::vec3 scale);

	glm::vec3 Forward();
	glm::vec3 Back();
	glm::vec3 Up();
	glm::vec3 Down();
	glm::vec3 Right();
	glm::vec3 Left();
private:
	glm::vec3 Rotate(glm::vec3 v, glm::quat q);

	glm::vec3 m_position;
	glm::quat m_rotation;
	glm::vec3 m_scale;
};

#endif
