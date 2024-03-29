# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /opt/cmake-3.16.1-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.16.1-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/homography_mixtures/SGridSearch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/homography_mixtures/SGridSearch

# Include any dependencies generated for this target.
include CMakeFiles/SGriSearch.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SGriSearch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SGriSearch.dir/flags.make

CMakeFiles/SGriSearch.dir/main.cpp.o: CMakeFiles/SGriSearch.dir/flags.make
CMakeFiles/SGriSearch.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/homography_mixtures/SGridSearch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SGriSearch.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SGriSearch.dir/main.cpp.o -c /data/homography_mixtures/SGridSearch/main.cpp

CMakeFiles/SGriSearch.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SGriSearch.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/homography_mixtures/SGridSearch/main.cpp > CMakeFiles/SGriSearch.dir/main.cpp.i

CMakeFiles/SGriSearch.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SGriSearch.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/homography_mixtures/SGridSearch/main.cpp -o CMakeFiles/SGriSearch.dir/main.cpp.s

CMakeFiles/SGriSearch.dir/SGridTracker.cpp.o: CMakeFiles/SGriSearch.dir/flags.make
CMakeFiles/SGriSearch.dir/SGridTracker.cpp.o: SGridTracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/homography_mixtures/SGridSearch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/SGriSearch.dir/SGridTracker.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SGriSearch.dir/SGridTracker.cpp.o -c /data/homography_mixtures/SGridSearch/SGridTracker.cpp

CMakeFiles/SGriSearch.dir/SGridTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SGriSearch.dir/SGridTracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/homography_mixtures/SGridSearch/SGridTracker.cpp > CMakeFiles/SGriSearch.dir/SGridTracker.cpp.i

CMakeFiles/SGriSearch.dir/SGridTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SGriSearch.dir/SGridTracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/homography_mixtures/SGridSearch/SGridTracker.cpp -o CMakeFiles/SGriSearch.dir/SGridTracker.cpp.s

# Object files for target SGriSearch
SGriSearch_OBJECTS = \
"CMakeFiles/SGriSearch.dir/main.cpp.o" \
"CMakeFiles/SGriSearch.dir/SGridTracker.cpp.o"

# External object files for target SGriSearch
SGriSearch_EXTERNAL_OBJECTS =

bin/SGriSearch: CMakeFiles/SGriSearch.dir/main.cpp.o
bin/SGriSearch: CMakeFiles/SGriSearch.dir/SGridTracker.cpp.o
bin/SGriSearch: CMakeFiles/SGriSearch.dir/build.make
bin/SGriSearch: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
bin/SGriSearch: /usr/local/lib/libopencv_dnn.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_gapi.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_highgui.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_ml.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_objdetect.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_photo.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_stitching.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_video.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_videoio.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_imgcodecs.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_calib3d.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_features2d.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_flann.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_imgproc.so.4.4.0
bin/SGriSearch: /usr/local/lib/libopencv_core.so.4.4.0
bin/SGriSearch: CMakeFiles/SGriSearch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/homography_mixtures/SGridSearch/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable bin/SGriSearch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SGriSearch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SGriSearch.dir/build: bin/SGriSearch

.PHONY : CMakeFiles/SGriSearch.dir/build

CMakeFiles/SGriSearch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SGriSearch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SGriSearch.dir/clean

CMakeFiles/SGriSearch.dir/depend:
	cd /data/homography_mixtures/SGridSearch && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/homography_mixtures/SGridSearch /data/homography_mixtures/SGridSearch /data/homography_mixtures/SGridSearch /data/homography_mixtures/SGridSearch /data/homography_mixtures/SGridSearch/CMakeFiles/SGriSearch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SGriSearch.dir/depend

