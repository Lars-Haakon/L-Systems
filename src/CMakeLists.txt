find_package (OpenGL REQUIRED)
find_package (GLEW REQUIRED)

#find glfw3
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

#set all source files
set (srcfiles main.cpp core/transform.cpp core/sceneobject.cpp core/camera.cpp rendering/mesh.cpp rendering/shader.cpp rendering/texture.cpp rendering/lsystem.cpp)

#set libraries
set (libs ${OPENGL_LIBRARY} ${GLEW_LIBRARY} ${GLFW_STATIC_LIBRARIES})

add_executable(${PROJECT_NAME} ${srcfiles})
target_link_libraries(${PROJECT_NAME} ${libs})
