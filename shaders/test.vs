#version 450

in layout(location=0) vec4 vertex;

uniform mat4 M;

void main()
{
	gl_Position = vec4((M*vec4(vertex.xyz, 1)).xyz, vertex.w);
}
