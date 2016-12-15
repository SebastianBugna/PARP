cmake_minimum_required(VERSION 2.8)

project( SCRATCHES )

find_package( OpenCV REQUIRED )

set(CMAKE_C_FLAGS " -g ")

#agrego sources manualmente
set(SOURCES Scratches.cpp Test_Scratches.cpp )

#creo ejecutable a partir de sources
add_executable( Test_Scratches ${SOURCES})


target_link_libraries( Test_Scratches ${OpenCV_LIBS} )
