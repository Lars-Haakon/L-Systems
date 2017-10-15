#version 450

#define PI 3.1415926535897932384626433832795
#define LOD 8

layout (lines) in;
layout (triangle_strip, max_vertices = 2*LOD) out;

uniform mat4 M;
uniform mat4 MVP;
uniform vec3 eye;

vec4 mul(vec4 q0, vec4 q1)
{
    float w = q0.w*q1.w - q0.x*q1.x - q0.y*q1.y - q0.z*q1.z;
    float x = q0.x*q1.w + q0.w*q1.x + q0.y*q1.z - q0.z*q1.y;
    float y = q0.y*q1.w + q0.w*q1.y + q0.z*q1.x - q0.x*q1.z;
    float z = q0.z*q1.w + q0.w*q1.z + q0.x*q1.y - q0.y*q1.x;

	return vec4(x, y, z, w);
}

vec4 conjugate(vec4 q)
{
    return vec4(-q.x, -q.y, -q.z, q.w);
}

vec4 angleAxis(vec3 axis, float angle)
{
    float sinHalfAngle = sin(angle / 2);
	float cosHalfAngle = cos(angle / 2);

	return vec4(   axis.x * sinHalfAngle,
    	           axis.y * sinHalfAngle,
                   axis.z * sinHalfAngle,
	               cosHalfAngle);
}

vec3 rotate(vec3 v, vec4 q)
{
    vec4 _q = conjugate(q);
    vec4 w = mul(mul(q, vec4(v, 0)), _q);

    return normalize(vec3(w.x, w.y, w.z));
}

vec3 forward(vec4 q)
{
    return rotate(vec3(0, 0, -1), q);
}

void main()
{
    vec4 worldPos0 = M*vec4(gl_in[0].gl_Position.xyz, 1);
    vec4 worldPos1 = M*vec4(gl_in[1].gl_Position.xyz, 1);

    vec3 v = normalize(worldPos0.xyz - worldPos1.xyz);

    for(int i = 0; i < LOD; i++)
    {
        float angle = ((2*PI)/(LOD-1))*i;
        vec4 q = angleAxis(v, angle);
        vec3 d = forward(q);

        gl_Position = MVP * vec4(worldPos0.xyz+d*gl_in[0].gl_Position.w, 1);
        EmitVertex();

        gl_Position = MVP * vec4(worldPos1.xyz+d*gl_in[1].gl_Position.w, 1);
        EmitVertex();
    }

    EndPrimitive();
}
