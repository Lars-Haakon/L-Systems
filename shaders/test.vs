#version 450

in layout(location=0) vec3 vertex;

uniform mat4 M;

void main()
{
	gl_Position = M*vec4(vertex, 1);
}
