#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include "transform.h"

class SceneObject
{
public:
	SceneObject(Transform transform = Transform());
	~SceneObject();

	Transform& GetTransform();
private:
	Transform m_transform;
};

#endif
