cmake_minimum_required(VERSION 3.10.2)
project(AV)
file(GLOB FILES_SRC
        "src/*.cpp"
        "src/*.c"
        "src/*.h"
        "include/Spline/*.cpp"
        "include/Spline/*.h"
        )

set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

# kalman headers
set(HEADERS
    # Base
    include/kalman/KalmanFilterBase.hpp
    include/kalman/StandardBase.hpp
    include/kalman/SquareRootBase.hpp
    include/kalman/StandardFilterBase.hpp
    include/kalman/SquareRootFilterBase.hpp

    # EKF
    include/kalman/ExtendedKalmanFilter.hpp
    include/kalman/SquareRootExtendedKalmanFilter.hpp

    # UKF
    include/kalman/UnscentedKalmanFilterBase.hpp
    include/kalman/UnscentedKalmanFilter.hpp
    include/kalman/SquareRootUnscentedKalmanFilter.hpp

    # Models
    include/kalman/MeasurementModel.hpp
    include/kalman/SystemModel.hpp
    include/kalman/LinearizedMeasurementModel.hpp
    include/kalman/LinearizedSystemModel.hpp

    # Utils
    include/kalman/Matrix.hpp
    include/kalman/Types.hpp
        src/SystemModel.h src/PositionMeasurementModel.h src/Clustering.cpp src/Clustering.h)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(include)
include_directories(res)
include_directories(/usr/local/include/eigen3)
include_directories(/usr/local/include/flann)
include_directories(/usr/local/include/flann)
include_directories(include/MathGeoLib)
include_directories(/home/andreas/Libraries/FreeImage3180/FreeImage)

### pcl
find_package(PCL 1.9 REQUIRED COMPONENTS common io)
#find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
### pcl

### json
set(JSON_DIR "${LIB_DIR}/json-develop")
add_subdirectory("${JSON_DIR}")
### json

### opencv
find_package( OpenCV REQUIRED )
### opencv

add_executable(AV ${FILES_SRC} ${HEADERS})
#add_executable(AV ${FILES_SRC} src/CurbDetection.cpp src/CurbDetection.h
#        src/Clustering.cpp src/Clustering.h ${HEADERS})

target_link_libraries(AV -lglfw3
                         -lMathGeoLib
                         -lpcl_kdtree
                         -lpcl_filters
                         -lpcl_segmentation
                         -lpcl_sample_consensus
                         -lpcl_search
                         -lpcl_ml
                         -lpcl_features
                         -lfreeimage
                         ${PCL_LIBRARIES} ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})

target_include_directories(AV PRIVATE "${JSON_DIR}/include" ${PCL_DEFINITIONS} ${PCL_INCLUDE_DIRS} ${PCL_LIBRARY_DIRS} ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS})