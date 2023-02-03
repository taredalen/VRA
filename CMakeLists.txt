cmake_minimum_required(VERSION 3.19)
project(VRA)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES "Tutorial 5" "Tutorial 5/template_ar_3d.cpp")
add_executable(TP ${SOURCE_FILES})

include_directories(${ZBAR_INCLUDE_DIR})


include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(TP ${OpenCV_LIBS})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
target_link_libraries(TP ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZBar REQUIRED IMPORTED_TARGET zbar)

target_link_libraries(TP PkgConfig::ZBar)