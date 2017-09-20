#ifndef CAMERA_H
#define CAMERA_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "sceneobject.h"

class Camera : public SceneObject
{
public:
	Camera(	Transform transform = Transform(),
        float speed = 10.0f,
        float sensitivity = 0.01f,
		float fov = 70.0f,
		float aspectRatio = 4.0f/3.0f,
		float near = 0.1f,
		float far = 100.0f);
	~Camera();

	glm::mat4 GetViewMatrix();
	glm::mat4 GetProjectionMatrix();
	void Move(glm::vec3 dir, float deltaTime);
	void Yaw(float xdelta);
	void Pitch(float ydelta);
private:
    float m_speed;
    float m_sensitivity;

	float m_fov;
	float m_aspectRatio;
	float m_near;
	float m_far;
};

#endif
