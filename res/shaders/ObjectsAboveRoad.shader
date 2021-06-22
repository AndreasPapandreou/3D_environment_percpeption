#shader vertex
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aCol;

out vec3 col;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    col = aCol;
}

    #shader fragment
    #version 330 core
out vec4 FragColor;
in vec3 col;

void main()
{
    FragColor = vec4(col, 1.0);
    //     FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}