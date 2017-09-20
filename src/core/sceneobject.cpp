#include "sceneobject.h"

SceneObject::SceneObject(Transform transform)
{
	m_transform = transform;
}

SceneObject::~SceneObject()
{

}

Transform& SceneObject::GetTransform()
{
	return m_transform;
}
