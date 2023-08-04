# cmake needs this line
cmake_minimum_required(VERSION 2.8) 
# Enable C++11 
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fPIC")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED TRUE) 
# Define project name 
project(try_inference) 
# Find OpenCV, you may need to set OpenCV_DIR variable 
# to the absolute path to the directory containing OpenCVConfig.cmake file 
# via the command line or GUI 
find_package(OpenCV REQUIRED) 
FIND_PACKAGE(PkgConfig REQUIRED) 
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system filesystem regex REQUIRED)

#find protobuf
INCLUDE(FindProtobuf)
find_package(Protobuf REQUIRED)
#find_package(Boost REQUIRED)

include_directories(include)

file(GLOB SOURCES)
file(GLOB HEADERS)

aux_source_directory(src DIR_SRCS)
# set the library dependencies.
set(LIB_DEPENDENCIES
   ${GLOG_LIB}
   ${OpenCV_LIBS}
   ${EIGEN3_LIBS}
   tensorflow_cc
   ${PROTOBUF_LIBRARY}
)

include_directories( 
  /usr/local/include/tf
  /usr/local/include/tf/tensorflow
  /usr/local/include/absl
  /usr/local/include/tf/third_party
${OpenCV_INCLUDE_DIRS}
 ${EIGEN3_INCLUDE_DIR}
)
 
link_directories(
  ${OpenCV_LIBRARY_DIRS}  
  
)

# Declare the executable target built from your sources 
#add_executable(edgeDetect rcf.cpp rcfTest.cpp guidedFilter.cpp) 
add_executable(${PROJECT_NAME} ${DIR_SRCS})
# Link your application with OpenCV libraries 
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${LIB_DEPENDENCIES})

