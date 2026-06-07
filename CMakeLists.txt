cmake_minimum_required(VERSION 3.20)
project(fMagCalc LANGUAGES Fortran)

# --- Standards / build type -------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g -fcheck=all -fbacktrace -Wall")

# --- Dependencies -----------------------------------------------------------
find_package(LAPACK REQUIRED)   # macOS: Accelerate; Linux: OpenBLAS/MKL
find_package(OpenMP)

# --- Core library -----------------------------------------------------------
add_library(magcalc_core
  src/magcalc_kinds.f90
  src/magcalc_diag.f90
  src/magcalc_kkd.f90
  src/magcalc_driver.f90
)
target_link_libraries(magcalc_core PUBLIC ${LAPACK_LIBRARIES})
if(OpenMP_Fortran_FOUND)
  target_link_libraries(magcalc_core PUBLIC OpenMP::OpenMP_Fortran)
endif()

# --- Tests ------------------------------------------------------------------
enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_diag.f90")
  add_executable(test_diag tests/test_diag.f90)
  target_link_libraries(test_diag PRIVATE magcalc_core)
  add_test(NAME diag COMMAND test_diag)
endif()
