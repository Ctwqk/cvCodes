/*
Bruce A. Maxwell
S21
Sample code to identify image fils in a directory


Taiwe Cui modified in 2024/2/1
*/
#include "readfile.h"
/*
Given a directory on the command line, scans through the directory for image
files.
Prints out the full path name for each file. This can be used as an argument to
fopen or to cv::imread.
*/

int listImgs(char dirname[256], std::vector<std::string> &images) {
	char buffer[256];
	FILE *fp;
	DIR *dirp;
	struct dirent *dp;
	int i;
	// check for sufficient arguments
	/*if( argc  < 2) {
		printf("usage: %s <directory path>\n", argv[0]);
		exit(-1);
	}
	*/
	// get the directory path
	printf("Processing directory %s\n", dirname );
	// open the directory
	dirp = opendir( dirname );
	if( dirp == NULL) {
		printf("Cannot open directory %s\n", dirname);
		exit(-1);
	}
	std::string tmp;
	// loop over all the files in the image file listing
	while( (dp = readdir(dirp)) != NULL ) {
		// check if the file is an image
		if( strstr(dp->d_name, ".jpg") ||
		strstr(dp->d_name, ".png") ||
		strstr(dp->d_name, ".ppm") ||
		strstr(dp->d_name, ".tif") ) {
			//printf("processing image file: %s\n", dp->d_name);
			// build the overall filename
			strcpy(buffer, dirname);
			strcat(buffer, "/");
			strcat(buffer, dp->d_name);
			tmp=buffer;
			images.push_back(tmp);
		}
		//printf("1 \n");
	}
	printf("Terminating\n");
	//for(char* x:images) printf("%s \n",x);
	return(0);
}
