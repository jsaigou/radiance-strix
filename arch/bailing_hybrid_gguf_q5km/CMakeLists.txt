# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S arch/bailing_hybrid_gguf_q5km -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(bailing_hybrid_gguf_q5km LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(bailing_hybrid_gguf_q5km architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bailing_hybrid_gguf_q5km.cpp)
