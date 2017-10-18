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
	float x = m_rotation[0];
	float y = m_rotation[1];
	float z = m_rotation[2];
	float w = m_rotation[3];

	return -glm::vec3(2*(x*z + w*y), 2*(y*z - w*x), 1-2*(x*x + y*y)); // Negate because -z is forward
}

glm::vec3 Transform::Back()
{
	return -Forward();
}

glm::vec3 Transform::Up()
{
	float x = m_rotation[0];
	float y = m_rotation[1];
	float z = m_rotation[2];
	float w = m_rotation[3];

	return glm::vec3(2*(x*y - w*z), 1-2*(x*x + z*z), 2*(y*z + w*x));
}

glm::vec3 Transform::Down()
{
	return -Up();
}

glm::vec3 Transform::Right()
{
	float x = m_rotation[0];
	float y = m_rotation[1];
	float z = m_rotation[2];
	float w = m_rotation[3];

	return glm::vec3(1-2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y));
}

glm::vec3 Transform::Left()
{
	return -Right();
}

glm::vec3 Transform::Rotate(glm::vec3 v, glm::quat q)
{
	glm::quat _q = glm::conjugate(q);
	glm::quat w = q * glm::quat(0, v) * _q;

	return glm::normalize(glm::vec3(w[0], w[1], w[2]));
}
