#version 450

#define PI 3.1415926535897932384626433832795
#define LOD 50

layout (lines) in;
layout (triangle_strip, max_vertices = 2*(LOD+1)) out;

out vec2 texel;

uniform mat4 VP;
uniform vec3 eye;

vec4 angleAxis(vec3 axis, float angle)
{
    float sinHalfAngle = sin(angle / 2.0);
	float cosHalfAngle = cos(angle / 2.0);

	return vec4(   axis.x * sinHalfAngle,
    	           axis.y * sinHalfAngle,
                   axis.z * sinHalfAngle,
	               cosHalfAngle);
}

void main()
{
    vec3 a = normalize(gl_in[0].gl_Position.xyz - gl_in[1].gl_Position.xyz);

    for(int i = 0; i <= LOD; i++)
    {
        float angle = ((2.0*PI)/LOD)*i;
        vec4 q = angleAxis(a, angle);
        vec3 b = vec3(1-2*(q.y*q.y + q.z*q.z), 2*(q.x*q.y + q.z*q.w), 2*(q.x*q.z - q.w*q.y));
        vec3 c = normalize(cross(b, a));

        texel = vec2(angle/(2.0*PI), 0);
        gl_Position = VP*vec4(gl_in[0].gl_Position.xyz+c*gl_in[0].gl_Position.w, 1);
        EmitVertex();

        texel = vec2(angle/(2.0*PI), 1);
        gl_Position = VP*vec4(gl_in[1].gl_Position.xyz+c*gl_in[1].gl_Position.w, 1);
        EmitVertex();
    }

    EndPrimitive();
}
