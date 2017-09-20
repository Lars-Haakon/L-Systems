find_package (OpenGL REQUIRED)
find_package (GLEW REQUIRED)

find_package (CUDA)
add_subdirectory(cuda)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

#set all source files
set (srcfiles main.cpp core/transform.cpp core/sceneobject.cpp core/camera.cpp rendering/mesh.cpp rendering/shader.cpp rendering/lsystem.cpp)

#set libraries
set (libs ${OPENGL_LIBRARY} ${GLEW_LIBRARY} ${GLFW_STATIC_LIBRARIES})

add_executable(${PROJECT_NAME} ${srcfiles})

target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_SOURCE_DIR}/src/cuda)
add_dependencies(${PROJECT_NAME} cudalib)

target_link_libraries(${PROJECT_NAME} cudalib ${libs})
