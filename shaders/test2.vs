#version 450

in layout(location=0) vec3 vertex;

uniform mat4 M;
uniform mat4 VP;

void main()
{
    vec4 t = M*vec4(vertex, 1);

	gl_Position = VP*t;
}
