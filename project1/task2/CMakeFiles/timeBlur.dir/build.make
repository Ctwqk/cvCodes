# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/taiwei/code/cvCode/project1/task2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/taiwei/code/cvCode/project1/task2

# Include any dependencies generated for this target.
include CMakeFiles/timeBlur.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/timeBlur.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/timeBlur.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/timeBlur.dir/flags.make

CMakeFiles/timeBlur.dir/timeBlur.cpp.o: CMakeFiles/timeBlur.dir/flags.make
CMakeFiles/timeBlur.dir/timeBlur.cpp.o: timeBlur.cpp
CMakeFiles/timeBlur.dir/timeBlur.cpp.o: CMakeFiles/timeBlur.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project1/task2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/timeBlur.dir/timeBlur.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/timeBlur.dir/timeBlur.cpp.o -MF CMakeFiles/timeBlur.dir/timeBlur.cpp.o.d -o CMakeFiles/timeBlur.dir/timeBlur.cpp.o -c /home/taiwei/code/cvCode/project1/task2/timeBlur.cpp

CMakeFiles/timeBlur.dir/timeBlur.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/timeBlur.dir/timeBlur.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project1/task2/timeBlur.cpp > CMakeFiles/timeBlur.dir/timeBlur.cpp.i

CMakeFiles/timeBlur.dir/timeBlur.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/timeBlur.dir/timeBlur.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project1/task2/timeBlur.cpp -o CMakeFiles/timeBlur.dir/timeBlur.cpp.s

CMakeFiles/timeBlur.dir/filter.cpp.o: CMakeFiles/timeBlur.dir/flags.make
CMakeFiles/timeBlur.dir/filter.cpp.o: filter.cpp
CMakeFiles/timeBlur.dir/filter.cpp.o: CMakeFiles/timeBlur.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project1/task2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/timeBlur.dir/filter.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/timeBlur.dir/filter.cpp.o -MF CMakeFiles/timeBlur.dir/filter.cpp.o.d -o CMakeFiles/timeBlur.dir/filter.cpp.o -c /home/taiwei/code/cvCode/project1/task2/filter.cpp

CMakeFiles/timeBlur.dir/filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/timeBlur.dir/filter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project1/task2/filter.cpp > CMakeFiles/timeBlur.dir/filter.cpp.i

CMakeFiles/timeBlur.dir/filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/timeBlur.dir/filter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project1/task2/filter.cpp -o CMakeFiles/timeBlur.dir/filter.cpp.s

# Object files for target timeBlur
timeBlur_OBJECTS = \
"CMakeFiles/timeBlur.dir/timeBlur.cpp.o" \
"CMakeFiles/timeBlur.dir/filter.cpp.o"

# External object files for target timeBlur
timeBlur_EXTERNAL_OBJECTS =

timeBlur: CMakeFiles/timeBlur.dir/timeBlur.cpp.o
timeBlur: CMakeFiles/timeBlur.dir/filter.cpp.o
timeBlur: CMakeFiles/timeBlur.dir/build.make
timeBlur: /usr/local/lib/libopencv_gapi.so.4.9.0
timeBlur: /usr/local/lib/libopencv_highgui.so.4.9.0
timeBlur: /usr/local/lib/libopencv_ml.so.4.9.0
timeBlur: /usr/local/lib/libopencv_objdetect.so.4.9.0
timeBlur: /usr/local/lib/libopencv_photo.so.4.9.0
timeBlur: /usr/local/lib/libopencv_stitching.so.4.9.0
timeBlur: /usr/local/lib/libopencv_video.so.4.9.0
timeBlur: /usr/local/lib/libopencv_videoio.so.4.9.0
timeBlur: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
timeBlur: /usr/local/lib/libopencv_dnn.so.4.9.0
timeBlur: /usr/local/lib/libopencv_calib3d.so.4.9.0
timeBlur: /usr/local/lib/libopencv_features2d.so.4.9.0
timeBlur: /usr/local/lib/libopencv_flann.so.4.9.0
timeBlur: /usr/local/lib/libopencv_imgproc.so.4.9.0
timeBlur: /usr/local/lib/libopencv_core.so.4.9.0
timeBlur: CMakeFiles/timeBlur.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/taiwei/code/cvCode/project1/task2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable timeBlur"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/timeBlur.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/timeBlur.dir/build: timeBlur
.PHONY : CMakeFiles/timeBlur.dir/build

CMakeFiles/timeBlur.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/timeBlur.dir/cmake_clean.cmake
.PHONY : CMakeFiles/timeBlur.dir/clean

CMakeFiles/timeBlur.dir/depend:
	cd /home/taiwei/code/cvCode/project1/task2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/taiwei/code/cvCode/project1/task2 /home/taiwei/code/cvCode/project1/task2 /home/taiwei/code/cvCode/project1/task2 /home/taiwei/code/cvCode/project1/task2 /home/taiwei/code/cvCode/project1/task2/CMakeFiles/timeBlur.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/timeBlur.dir/depend

