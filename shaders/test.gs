#version 450

layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 MVP;
uniform vec3 eye;

void main()
{
    vec3 d = normalize(cross(eye-gl_in[1].gl_Position.xyz, gl_in[0].gl_Position.xyz-gl_in[1].gl_Position.xyz));
    float l = 0.02f;

    gl_Position = MVP * vec4(gl_in[0].gl_Position.xyz+d*-gl_in[0].gl_Position.w, 1);
    EmitVertex();

    gl_Position = MVP * vec4(gl_in[0].gl_Position.xyz+d*gl_in[0].gl_Position.w, 1);
    EmitVertex();

    gl_Position = MVP * vec4(gl_in[1].gl_Position.xyz+d*-gl_in[1].gl_Position.w, 1);
    EmitVertex();

    gl_Position = MVP * vec4(gl_in[1].gl_Position.xyz+d*gl_in[1].gl_Position.w, 1);
    EmitVertex();

    EndPrimitive();
}
