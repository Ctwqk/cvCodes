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
CMAKE_SOURCE_DIR = /home/taiwei/code/cvCode/project3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/taiwei/code/cvCode/project3

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.cpp.o: main.cpp
CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c /home/taiwei/code/cvCode/project3/main.cpp

CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/main.cpp > CMakeFiles/main.dir/main.cpp.i

CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/main.cpp -o CMakeFiles/main.dir/main.cpp.s

CMakeFiles/main.dir/threshold.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/threshold.cpp.o: threshold.cpp
CMakeFiles/main.dir/threshold.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/threshold.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/threshold.cpp.o -MF CMakeFiles/main.dir/threshold.cpp.o.d -o CMakeFiles/main.dir/threshold.cpp.o -c /home/taiwei/code/cvCode/project3/threshold.cpp

CMakeFiles/main.dir/threshold.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/threshold.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/threshold.cpp > CMakeFiles/main.dir/threshold.cpp.i

CMakeFiles/main.dir/threshold.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/threshold.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/threshold.cpp -o CMakeFiles/main.dir/threshold.cpp.s

CMakeFiles/main.dir/denoise.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/denoise.cpp.o: denoise.cpp
CMakeFiles/main.dir/denoise.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/denoise.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/denoise.cpp.o -MF CMakeFiles/main.dir/denoise.cpp.o.d -o CMakeFiles/main.dir/denoise.cpp.o -c /home/taiwei/code/cvCode/project3/denoise.cpp

CMakeFiles/main.dir/denoise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/denoise.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/denoise.cpp > CMakeFiles/main.dir/denoise.cpp.i

CMakeFiles/main.dir/denoise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/denoise.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/denoise.cpp -o CMakeFiles/main.dir/denoise.cpp.s

CMakeFiles/main.dir/region.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/region.cpp.o: region.cpp
CMakeFiles/main.dir/region.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/main.dir/region.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/region.cpp.o -MF CMakeFiles/main.dir/region.cpp.o.d -o CMakeFiles/main.dir/region.cpp.o -c /home/taiwei/code/cvCode/project3/region.cpp

CMakeFiles/main.dir/region.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/region.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/region.cpp > CMakeFiles/main.dir/region.cpp.i

CMakeFiles/main.dir/region.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/region.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/region.cpp -o CMakeFiles/main.dir/region.cpp.s

CMakeFiles/main.dir/features.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/features.cpp.o: features.cpp
CMakeFiles/main.dir/features.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/main.dir/features.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/features.cpp.o -MF CMakeFiles/main.dir/features.cpp.o.d -o CMakeFiles/main.dir/features.cpp.o -c /home/taiwei/code/cvCode/project3/features.cpp

CMakeFiles/main.dir/features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/features.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/features.cpp > CMakeFiles/main.dir/features.cpp.i

CMakeFiles/main.dir/features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/features.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/features.cpp -o CMakeFiles/main.dir/features.cpp.s

CMakeFiles/main.dir/csv_util.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/csv_util.cpp.o: csv_util.cpp
CMakeFiles/main.dir/csv_util.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/main.dir/csv_util.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/csv_util.cpp.o -MF CMakeFiles/main.dir/csv_util.cpp.o.d -o CMakeFiles/main.dir/csv_util.cpp.o -c /home/taiwei/code/cvCode/project3/csv_util.cpp

CMakeFiles/main.dir/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/csv_util.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/csv_util.cpp > CMakeFiles/main.dir/csv_util.cpp.i

CMakeFiles/main.dir/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/csv_util.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/csv_util.cpp -o CMakeFiles/main.dir/csv_util.cpp.s

CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o: NetWorkThings/dnnEmbedding.cpp
CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o -MF CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o.d -o CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o -c /home/taiwei/code/cvCode/project3/NetWorkThings/dnnEmbedding.cpp

CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/NetWorkThings/dnnEmbedding.cpp > CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.i

CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/NetWorkThings/dnnEmbedding.cpp -o CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o" \
"CMakeFiles/main.dir/threshold.cpp.o" \
"CMakeFiles/main.dir/denoise.cpp.o" \
"CMakeFiles/main.dir/region.cpp.o" \
"CMakeFiles/main.dir/features.cpp.o" \
"CMakeFiles/main.dir/csv_util.cpp.o" \
"CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/main.cpp.o
main: CMakeFiles/main.dir/threshold.cpp.o
main: CMakeFiles/main.dir/denoise.cpp.o
main: CMakeFiles/main.dir/region.cpp.o
main: CMakeFiles/main.dir/features.cpp.o
main: CMakeFiles/main.dir/csv_util.cpp.o
main: CMakeFiles/main.dir/NetWorkThings/dnnEmbedding.cpp.o
main: CMakeFiles/main.dir/build.make
main: /usr/local/lib/libopencv_gapi.so.4.9.0
main: /usr/local/lib/libopencv_highgui.so.4.9.0
main: /usr/local/lib/libopencv_ml.so.4.9.0
main: /usr/local/lib/libopencv_objdetect.so.4.9.0
main: /usr/local/lib/libopencv_photo.so.4.9.0
main: /usr/local/lib/libopencv_stitching.so.4.9.0
main: /usr/local/lib/libopencv_video.so.4.9.0
main: /usr/local/lib/libopencv_videoio.so.4.9.0
main: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
main: /usr/local/lib/libopencv_dnn.so.4.9.0
main: /usr/local/lib/libopencv_calib3d.so.4.9.0
main: /usr/local/lib/libopencv_features2d.so.4.9.0
main: /usr/local/lib/libopencv_flann.so.4.9.0
main: /usr/local/lib/libopencv_imgproc.so.4.9.0
main: /usr/local/lib/libopencv_core.so.4.9.0
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/taiwei/code/cvCode/project3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

