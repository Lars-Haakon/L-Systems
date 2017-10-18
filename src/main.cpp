#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <sys/time.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "core/camera.h"
#include "rendering/mesh.h"
#include "rendering/shader.h"
#include "rendering/texture.h"
#include "rendering/lsystem.h"

void error_callback(int error, const char* description)
{
    printf("GLFW Error: %s\n", description);
}

void printOpenGLInfo()
{
    printf("Vendor: %s\n", glGetString(GL_VENDOR));
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
}

#define WIDTH 1600
#define HEIGHT 900

#define NUM_KEYS 350
bool keys[NUM_KEYS];

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(key != GLFW_KEY_UNKNOWN)
    {
        if(action == GLFW_PRESS)
        keys[key] = true;
        else if(action == GLFW_RELEASE)
        keys[key] = false;
    }
}

/*static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
float xdelta =  WIDTH/2 - (float)xpos;
cam.Yaw(xdelta);
float ydelta = HEIGHT/2 - (float)ypos;
cam.Pitch(ydelta);

glfwSetCursorPos(window, WIDTH/2, HEIGHT/2);
}*/

int main()
{
    if (!glfwInit())
    {
        return -1;
    }

    glfwSetErrorCallback(error_callback);

    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "3DOpenGL", NULL, NULL);
    if(!window)
    {
        glfwTerminate();
        return -1;
    }
    //glfwSetWindowPos(window, mode->width/2, mode->height/2);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    for(int i = 0; i < NUM_KEYS; i++)
    keys[i] = false;
    glfwSetKeyCallback(window, key_callback);

    //glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetCursorPos(window, WIDTH/2, HEIGHT/2);

    //glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK)
    {
        glfwTerminate();
        return -1;
    }
    printOpenGLInfo();

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_DEPTH_CLAMP);
    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);

    // camera setup
    Camera cam(Transform(glm::vec3(0, 2.0f, 0.0f), glm::normalize(glm::quat(1, 0, 0, 0))), .5f, 0.01f, 70.0f, WIDTH / (float) HEIGHT, 0.1f, 100.0f);

    double start = glfwGetTime();
    /*LSystem lSystem("F-F-F-F", 0.1f, 90.0f);
    lSystem.AddProduction('F', "F-F+F+FF-F-F+F");
    lSystem.Generate(2);*/

    /*LSystem lSystem("F+F+F+F", 0.5f, 90.0f);
    lSystem.AddProduction('F', "F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF");
    lSystem.AddProduction('f', "ffffff");
    lSystem.Generate(2);*/

    /*LSystem lSystem("X", 0.1f, 22.5f);
    lSystem.AddProduction('X', "F-[/[X]+X]+F[&+FX]-X");
    lSystem.AddProduction('F', "FF");
    lSystem.Generate(5);*/

    /*LSystem lSystem("F", 0.1f, 22.5f);
    lSystem.AddProduction('F', "FF-[-F+F+F]+[+F-F-F]");
    lSystem.Generate(4);*/

    // Tree
    LSystem lSystem("FFFA", 0.1f, 25.0f);
    lSystem.AddProduction('A', "[B]////[B]////[B]");
    lSystem.AddProduction('B', "&FFFA");
    lSystem.Generate(10);

    // Hilbert Curve
    /*LSystem lSystem("A", 0.5f, 90.0f);
    lSystem.AddProduction('A', "B-F+CFC+F-D&F^D-F+&&CFC+F+B//");
    lSystem.AddProduction('B', "A&F^CFB^F^D^^-F-D^|F^B|FC^F^A//");
    lSystem.AddProduction('C', "|D^|F^B-F+C^F^A&&FA&F^C+F+B^F^D//");
    lSystem.AddProduction('D', "|CFB-F+B|FA&F^A&&FB-F+B|FC//");
    lSystem.Generate(3);*/

    double end = glfwGetTime();
    printf("Execution time: %.5f\n", end-start);

    Shader shader;
    shader.AddVertexShader("test.vs");
    shader.AddGeometryShader("cylinder.gs");
    shader.AddFragmentShader("test.fs");
    shader.CompileShader();
    int MLocation = shader.AddUniform("M");
    int VPLocation = shader.AddUniform("VP");
    int eyeLocation = shader.AddUniform("eye");
    int diffuseLocation = shader.AddUniform("diffuseMap");

    Texture t("tre4.bmp");

    float lastTime = (float) glfwGetTime();
    float passedTime = 0.0f;
    int frameCount = 0;

    while (!glfwWindowShouldClose(window))
    {
        float newTime = (float) glfwGetTime();
        float deltaTime = newTime - lastTime;
        lastTime = newTime;

        frameCount++;
        passedTime += deltaTime;
        if(passedTime > 1.0)
        {
            printf("FPS: %d\n", frameCount);

            passedTime = 0.0f;
            frameCount = 0;
        }

        // input
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        float xdelta =  WIDTH/2 - (float)xpos;
        cam.Yaw(xdelta);
        float ydelta = HEIGHT/2 - (float)ypos;
        cam.Pitch(ydelta);

        glfwSetCursorPos(window, WIDTH/2, HEIGHT/2);

        if(keys[GLFW_KEY_W])
            cam.Move(cam.GetTransform().Forward(), deltaTime);
        if(keys[GLFW_KEY_S])
            cam.Move(cam.GetTransform().Back(), deltaTime);
        if(keys[GLFW_KEY_D])
            cam.Move(cam.GetTransform().Right(), deltaTime);
        if(keys[GLFW_KEY_A])
            cam.Move(cam.GetTransform().Left(), deltaTime);

        // update
        glm::mat4 modelMatrix = lSystem.GetTransform().GetModelMatrix();

        glm::mat4 viewMatrix = cam.GetViewMatrix();
        glm::mat4 projectionMatrix = cam.GetProjectionMatrix();

        // render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.Bind();
        shader.SetUniformMat4(MLocation, glm::value_ptr(modelMatrix));
        shader.SetUniformMat4(VPLocation, glm::value_ptr(projectionMatrix * viewMatrix));
        shader.SetUniformVec3(eyeLocation, cam.GetTransform().GetPosition()[0], cam.GetTransform().GetPosition()[1], cam.GetTransform().GetPosition()[2]);

        glActiveTexture(GL_TEXTURE0);
        t.Bind();
        glUniform1i(diffuseLocation, 0);

        lSystem.Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
