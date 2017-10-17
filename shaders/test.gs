#version 450

layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 VP;
uniform vec3 eye;

void main()
{
    vec4 worldPos0 = vec4(gl_in[0].gl_Position.xyz, 1);
    vec4 worldPos1 = vec4(gl_in[1].gl_Position.xyz, 1);

    vec3 d = normalize(cross(eye-worldPos1.xyz, worldPos0.xyz-worldPos1.xyz));

    gl_Position = VP * vec4(worldPos0.xyz+d*-gl_in[0].gl_Position.w, 1);
    EmitVertex();

    gl_Position = VP * vec4(worldPos0.xyz+d*gl_in[0].gl_Position.w, 1);
    EmitVertex();

    gl_Position = VP * vec4(worldPos1.xyz+d*-gl_in[1].gl_Position.w, 1);
    EmitVertex();

    gl_Position = VP * vec4(worldPos1.xyz+d*gl_in[1].gl_Position.w, 1);
    EmitVertex();

    EndPrimitive();
}
