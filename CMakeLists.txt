cmake_minimum_required(VERSION 3.19)
project(VRA)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PkgConfig REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES "Mini Projet" "Mini Projet/main.cpp")
add_executable(TP ${SOURCE_FILES})

include_directories(${ZBAR_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
target_link_libraries(TP ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

pkg_check_modules(ZBar REQUIRED IMPORTED_TARGET zbar)

target_link_libraries(TP PkgConfig::ZBar)