cmake_minimum_required(VERSION 3.14)

project(SNN VERSION 0.1 LANGUAGES CXX)
option(ENABLE_DOCTESTS "Include tests in the library. Setting this to OFF will remove all doctest related code.
                        Tests in tests/*.cpp will still be enabled." ON)


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(Doctest)


set(SOURCES
    src/neuron.cpp
    src/receptive_field.cpp
    src/spike_train.cpp
    src/utils.cpp
    src/learning.cpp
    src/network.cpp
    src/threshold.cpp)

set(TESTFILES
    tests/main.cpp)


set(LIBRARY_NAME engine) # default name for the library built from src/*.cpp
add_library(${LIBRARY_NAME} OBJECT ${SOURCES})
# Lib needs its header files, and users of the library must also see these (PUBLIC). (No change needed)
target_include_directories(${LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
# There's also (probably) doctests within the library, so we need to see this as well.
target_link_libraries(${LIBRARY_NAME} PUBLIC doctest)



add_executable(snn app/main.cpp)
target_link_libraries( snn PUBLIC ${OpenCV_LIBS} )
target_link_libraries( snn PRIVATE ${LIBRARY_NAME} )

set_target_properties(
    ${LIBRARY_NAME} snn
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_subdirectory(tests)
