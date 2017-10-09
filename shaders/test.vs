#version 450

in layout(location=0) vec4 vertex;

void main()
{
	gl_Position = vertex;
}
