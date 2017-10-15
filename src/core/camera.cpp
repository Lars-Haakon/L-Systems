#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <stdio.h>

Camera::Camera(Transform transform, float speed, float sensitivity,
                                    float fov, float aspectRatio, float near, float far)
: SceneObject(transform)
{
    m_speed = speed;
    m_sensitivity = sensitivity;

    m_fov = fov;
	m_aspectRatio = aspectRatio;
	m_near = near;
    m_far = far;


    /*Transform t(glm::vec3(0, 0, 0), glm::angleAxis(3.14f/2, glm::vec3(0, 1, 0)));
    glm::vec3 f = t.Forward();

    printf("%f %f %f\n", f[0], f[1], f[2]);*/
}

Camera::~Camera()
{

}

glm::mat4 Camera::GetViewMatrix()
{
	return glm::mat4_cast(glm::conjugate(GetTransform().GetRotation())) * glm::translate(glm::mat4(1.0f), -GetTransform().GetPosition());
}

glm::mat4 Camera::GetProjectionMatrix()
{
	return glm::perspective(
	    		glm::radians(m_fov), // The vertical Field of View, in radians: the amount of "zoom". Think "camera lens". Usually between 90° (extra wide) and 30° (quite zoomed in)
		    	m_aspectRatio,       // Aspect Ratio. Depends on the size of your window. Notice that 4/3 == 800/600 == 1280/960, sounds familiar ?
		    	m_near,              // Near clipping plane. Keep as big as possible, or you'll get precision issues.
		    	m_far);             // Far clipping plane. Keep as little as possible.
}

void Camera::Move(glm::vec3 dir, float deltaTime)
{
	GetTransform().SetPosition(GetTransform().GetPosition() + dir*m_speed*deltaTime);
}

void Camera::Yaw(float xdelta)
{
	GetTransform().SetRotation(glm::angleAxis(xdelta*m_sensitivity, glm::vec3(0, 1, 0)) * GetTransform().GetRotation());
}

void Camera::Pitch(float ydelta)
{
	GetTransform().SetRotation(glm::angleAxis(ydelta*m_sensitivity, GetTransform().Right()) * GetTransform().GetRotation());
}
