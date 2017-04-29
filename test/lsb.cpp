#include <iostream>
#include <string>

#include <cstdlib>
#include <climits>

#include <glob.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lsb.hpp"

std::vector<std::string> glob(const std::string& pat){
	glob_t glob_result;
	glob(pat.c_str(), GLOB_TILDE, NULL, &glob_result);

	std::vector<std::string> v;
	for(unsigned int i=0; i<glob_result.gl_pathc; i++)
		v.push_back(std::string(glob_result.gl_pathv[i]));

	globfree(&glob_result);
	return v;
}

std::vector<char> generate_data(int n)
{
	std::vector<char> v;
	for(int i = 0; i<n ; i++)
		v.push_back(rand()%(UCHAR_MAX+1));

	return v;
}

void test_grayscale(
	const std::vector<std::string>& image_path_list)
{
	for( const std::string& f : image_path_list ){

		std::cout << "Open the cover: " << f << std::endl;

		cv::Mat cover = cv::imread(f, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat stego;

		if(cover.data == nullptr){
			std::cerr << "Cannot open " << f << std::endl;
			exit(EXIT_FAILURE);
		}

		int max_bytes = (cover.rows*cover.cols)/CHAR_BIT + 1;
		int n_pixel_modified = rand()%max_bytes;
		std::vector<char> data = generate_data(n_pixel_modified);

		std::vector<char> extracted_data;

		stegim::lsb_embed(cover, stego, data);
		stegim::lsb_extract(stego, extracted_data, 0, data.size());

		if(data != extracted_data){
			std::cout << "Embeded (" << data.size() << "): " << std::endl;
			std::cout << data.data() << std::endl;
			std::cout << "Extracted (" << data.size() << "): " << std::endl;
			std::cout << extracted_data.data() << std::endl;
			std::cerr << "Extracted data is different from embedded data!"
				  << std::endl;
			exit(EXIT_FAILURE);
		}

	}
}

int main()
{

	std::string cover_image_path(COVER_IMAGE_PATH);

	test_grayscale(glob(cover_image_path + "/*.pgm"));

	return 0;
}
