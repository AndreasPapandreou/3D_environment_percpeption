# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/andreas/IDEs/CLion-2019.1.4/clion-2019.1.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/andreas/IDEs/CLion-2019.1.4/clion-2019.1.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/andreas/Storage/kalman-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/andreas/Storage/kalman-master

# Include any dependencies generated for this target.
include external/googletest/CMakeFiles/gtest_main.dir/depend.make

# Include the progress variables for this target.
include external/googletest/CMakeFiles/gtest_main.dir/progress.make

# Include the compile flags for this target's objects.
include external/googletest/CMakeFiles/gtest_main.dir/flags.make

external/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o: external/googletest/CMakeFiles/gtest_main.dir/flags.make
external/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o: external/googletest/src/gtest_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/andreas/Storage/kalman-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o"
	cd /media/andreas/Storage/kalman-master/external/googletest && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gtest_main.dir/src/gtest_main.cc.o -c /media/andreas/Storage/kalman-master/external/googletest/src/gtest_main.cc

external/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gtest_main.dir/src/gtest_main.cc.i"
	cd /media/andreas/Storage/kalman-master/external/googletest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/andreas/Storage/kalman-master/external/googletest/src/gtest_main.cc > CMakeFiles/gtest_main.dir/src/gtest_main.cc.i

external/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gtest_main.dir/src/gtest_main.cc.s"
	cd /media/andreas/Storage/kalman-master/external/googletest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/andreas/Storage/kalman-master/external/googletest/src/gtest_main.cc -o CMakeFiles/gtest_main.dir/src/gtest_main.cc.s

# Object files for target gtest_main
gtest_main_OBJECTS = \
"CMakeFiles/gtest_main.dir/src/gtest_main.cc.o"

# External object files for target gtest_main
gtest_main_EXTERNAL_OBJECTS =

external/googletest/libgtest_main.a: external/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o
external/googletest/libgtest_main.a: external/googletest/CMakeFiles/gtest_main.dir/build.make
external/googletest/libgtest_main.a: external/googletest/CMakeFiles/gtest_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/andreas/Storage/kalman-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libgtest_main.a"
	cd /media/andreas/Storage/kalman-master/external/googletest && $(CMAKE_COMMAND) -P CMakeFiles/gtest_main.dir/cmake_clean_target.cmake
	cd /media/andreas/Storage/kalman-master/external/googletest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gtest_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/googletest/CMakeFiles/gtest_main.dir/build: external/googletest/libgtest_main.a

.PHONY : external/googletest/CMakeFiles/gtest_main.dir/build

external/googletest/CMakeFiles/gtest_main.dir/clean:
	cd /media/andreas/Storage/kalman-master/external/googletest && $(CMAKE_COMMAND) -P CMakeFiles/gtest_main.dir/cmake_clean.cmake
.PHONY : external/googletest/CMakeFiles/gtest_main.dir/clean

external/googletest/CMakeFiles/gtest_main.dir/depend:
	cd /media/andreas/Storage/kalman-master && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/andreas/Storage/kalman-master /media/andreas/Storage/kalman-master/external/googletest /media/andreas/Storage/kalman-master /media/andreas/Storage/kalman-master/external/googletest /media/andreas/Storage/kalman-master/external/googletest/CMakeFiles/gtest_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/googletest/CMakeFiles/gtest_main.dir/depend

