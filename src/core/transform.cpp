#include <glm/gtc/matrix_transform.hpp>

#include "transform.h"

Transform::Transform(glm::vec3 position, glm::quat rotation, glm::vec3 scale)
{
	m_position = position;
	m_rotation = rotation;
    m_scale = scale;
}

Transform::~Transform()
{

}

glm::mat4 Transform::GetModelMatrix()
{
	glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), m_position);
	glm::mat4 rotationMatrix = glm::mat4_cast(m_rotation);
	glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), m_scale);

	return translationMatrix * rotationMatrix * scaleMatrix;
}

glm::vec3 Transform::GetPosition()
{
	return m_position;
}

void Transform::SetPosition(glm::vec3 position)
{
	m_position = position;
}

glm::quat Transform::GetRotation()
{
	return m_rotation;
}

void Transform::SetRotation(glm::quat rotation)
{
	m_rotation = rotation;
}

glm::vec3 Transform::GetScale()
{
	return m_scale;
}

void Transform::SetScale(glm::vec3 scale)
{
	m_scale = scale;
}

glm::vec3 Transform::Forward()
{
	return Rotate(glm::vec3(0, 0, -1), m_rotation);
}

glm::vec3 Transform::Back()
{
	return Rotate(glm::vec3(0, 0, 1), m_rotation);
}

glm::vec3 Transform::Up()
{
	return Rotate(glm::vec3(0, 1, 0), m_rotation);
}

glm::vec3 Transform::Down()
{
	return Rotate(glm::vec3(0, -1, 0), m_rotation);
}

glm::vec3 Transform::Right()
{
	return Rotate(glm::vec3(1, 0, 0), m_rotation);
}

glm::vec3 Transform::Left()
{
	return Rotate(glm::vec3(-1, 0, 0), m_rotation);
}

glm::vec3 Transform::Rotate(glm::vec3 v, glm::quat q)
{
	glm::quat _q = glm::conjugate(q);
	glm::quat w = q * glm::quat(0, v) * _q;

	return glm::normalize(glm::vec3(w[0], w[1], w[2]));
}
