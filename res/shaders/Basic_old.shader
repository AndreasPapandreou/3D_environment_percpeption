#shader vertex
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

// out VS_OUT {
//     vec3 color;
// } vs_out;

out vec3 color;

void main()
{
//     vs_out.color = aColor;
    color = aColor;
    gl_Position = vec4(aPos.x, aPos.y, 1.0, 1.0);
}

#shader geometry
#version 330 core
layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

in VS_OUT {
    vec3 color;
} gs_in[];

out vec3 fColor;

void build_house(vec4 position)
{
    fColor = gs_in[0].color; // gs_in[0] since there's only one input vertex
    gl_Position = position + vec4(-0.1, -0.1, 0.0, 0.0); // 1:bottom-left
    EmitVertex();
    gl_Position = position + vec4( 0.1, -0.1, 0.0, 0.0); // 2:bottom-right
    EmitVertex();
    EndPrimitive();
}

void main() {
    build_house(gl_in[0].gl_Position);
}

#shader fragment
#version 330 core
out vec4 FragColor;

in vec3 color;

void main()
{
    FragColor = vec4(color, 1.0);
}


*******************************************************

#shader vertex
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aOffset;

out vec3 fColor;

void main()
{
    fColor = aColor;
    gl_Position = vec4(aPos + aOffset, 0.0, 1.0);
}

#shader fragment
#version 330 core
out vec4 FragColor;

in vec3 fColor;

void main()
{
    FragColor = vec4(fColor, 1.0);
}
