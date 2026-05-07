cmake_minimum_required(VERSION 3.10)
project(RoboMaster_Assignment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV 4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(calibrate src/calibration/calibrate.cpp)
target_link_libraries(calibrate ${OpenCV_LIBS})

add_executable(armor_detect src/detection/armor_detect.cpp)
target_link_libraries(armor_detect ${OpenCV_LIBS})

add_executable(pnp src/pnp/pose_estimation.cpp)
target_link_libraries(pnp ${OpenCV_LIBS})

add_executable(armor_video src/detection/armor_video.cpp)
target_link_libraries(armor_video ${OpenCV_LIBS})
