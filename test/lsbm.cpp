#include <iostream>
#include <sstream>

#include <glob.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lsb_matching.hpp"

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

void test_grayscale(std::vector<std::string>& image_path_list)
{
	size_t n_img = 0;
	for(std::string& path : image_path_list){
		std::cout << "File: " << path << std::endl;
		cv::Mat cover = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat stego;

		size_t max_bytes = (cover.rows*cover.cols)/CHAR_BIT;
		size_t n_data = rand()%(max_bytes + 1);
		std::vector<char> data = generate_data(n_data);

		std::vector<char> key = generate_data(10);
		std::vector<char> extracted_data;

		stegim::lsb_matching_embed(cover, stego, data, key);
		stegim::lsb_matching_extract(stego, extracted_data, data.size(), key);

		if(n_img < 5){

			std::stringstream fcover;
			std::stringstream fstego;


			fcover << n_img << "_lsbm_cover.pgm";
			fstego << n_img << "_lsbm_stego.pgm";

			cv::imwrite(fcover.str(), cover);
			cv::imwrite(fstego.str(), stego);
		}

		if(data != extracted_data){
			std::cout << "Data is different from extrated!" << std::endl;
			exit(EXIT_FAILURE);
		}

		n_img++;
	}
}

int main()
{

	std::string cover_image_path(COVER_IMAGE_PATH);
	std::string img_path = cover_image_path;
	std::vector<std::string> image_path_list = glob(img_path + "/*.pgm");

	test_grayscale(image_path_list);

	return 0;
}
