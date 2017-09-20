cmake_minimum_required (VERSION 2.8)
project (3DOpenGL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
subdirs (src)

file(COPY shaders DESTINATION .)
