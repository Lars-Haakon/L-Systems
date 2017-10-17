#version 450

in vec2 texel;

out vec4 fragColor;

uniform sampler2D diffuseMap;

void main()
{
	fragColor = texture2D(diffuseMap, texel);
}
