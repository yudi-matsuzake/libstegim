#include <iostream>
#include <string>

#include <sstream>

#include <cstdlib>
#include <ctime>
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

void print_data(std::vector<char>& v)
{
	if(v.size() <= 20){
		for(size_t i=0; i<v.size(); i++)
			std::cout << "[" << ((int)v[i]) << "]";
	}else{
		for(int i=0; i<10; i++)
			std::cout << "[" << ((int)v[i]) << "]";

		std::cout << "...";
		
		for(size_t i=v.size() - 10 - 1; i<v.size(); i++)
			std::cout << "[" << ((int)v[i]) << "]";
	}

	std::cout << std::endl;
}

void test_grayscale(
	const std::vector<std::string>& image_path_list)
{
	int first = 5;
	int n = 0;
	for( const std::string& f : image_path_list ){

		cv::Mat cover = cv::imread(f, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat stego;

		if(cover.data == nullptr){
			std::cerr << "Cannot open " << f << std::endl;
			exit(EXIT_FAILURE);
		}

		/*
		 * generate random data to embed
		 */
		int max_bytes = (cover.rows*cover.cols)/CHAR_BIT;
		int data_size = rand()%(max_bytes + 1);
		std::vector<char> data = generate_data(data_size);

		std::vector<char> extracted_data;

		/*
		 * set offset
		 */
		stegim::lsb_options lsb_opt;
		int offset_range = max_bytes - data_size;
		int offset;

		if(offset_range > 0)
			offset = rand()%offset_range;
		else
			offset = 0;

		lsb_opt.set_offset(offset);

		/*
		 * print embed informations
		 */
		std::cout
			<< "File: " << f << std::endl
			<< "N pixel: " << data_size << std::endl
			<< "Offset: " << offset << std::endl;

		/*
		 * embed
		 */
		stegim::lsb_embed(cover, stego, data, lsb_opt);
		stegim::lsb_extract(stego, extracted_data, data.size(), lsb_opt);

		/*
		 * write the first cover and stego images
		 */
		if(n < first){
			std::stringstream fcover;
			std::stringstream fstego;
			fcover << "lsb_cover_" << n << ".pgm";
			fstego << "lsb_stego_" << n << ".pgm";
			n++;
			cv::imwrite(fcover.str(), cover);
			cv::imwrite(fstego.str(), stego);
		}

		/*
		 * test the extraction
		 */
		if(data != extracted_data){
			std::cout << "Embeded (" << data.size() << "): " << std::endl;
			print_data(data);
			std::cout << "Extracted (" << data.size() << "): " << std::endl;
			print_data(extracted_data);
			std::cerr << "Extracted data is different from embedded data!"
				  << std::endl;
			exit(EXIT_FAILURE);
		}

	}
}

void test_color(
	const std::vector<std::string>& image_path_list)
{
	int first = 5;
	int n = 0;
	for( const std::string& f : image_path_list ){

		cv::Mat cover = cv::imread(f, CV_LOAD_IMAGE_COLOR);
		cv::Mat stego;

		if(cover.data == nullptr){
			std::cerr << "Cannot open " << f << std::endl;
			exit(EXIT_FAILURE);
		}

		stegim::lsb_options lsb_opt;

		lsb_opt	.set_b(rand()%2)
			.set_g(rand()%2)
			.set_r(rand()%2);

		if(!lsb_opt.get_b() && !lsb_opt.get_g() && !lsb_opt.get_r())
			lsb_opt.set_b(true);

		int n_channels = lsb_opt.get_b() + lsb_opt.get_g() + lsb_opt.get_r();

		/*
		 * generate random data to embed
		 */
		int max_bytes = (cover.rows*cover.cols*n_channels)/CHAR_BIT;
		int data_size = rand()%(max_bytes + 1);
		std::vector<char> data = generate_data(data_size);

		std::vector<char> extracted_data;

		/*
		 * set offset
		 */
		int offset_range = max_bytes - data_size;
		int offset;

		if(offset_range > 1)
			offset = rand()%offset_range;
		else
			offset = 0;

		lsb_opt.set_offset(offset);

		/*
		 * print embed informations
		 */
		std::cout
			<< "File: " << f << std::endl
			<< "N pixel: " << data_size << std::endl
			<< "Offset: " << offset << std::endl
			<< "B: " << lsb_opt.get_b() << std::endl
			<< "G: " << lsb_opt.get_g() << std::endl
			<< "R: " << lsb_opt.get_r() << std::endl;

		/*
		 * embed
		 */
		stegim::lsb_embed(cover, stego, data, lsb_opt);
		stegim::lsb_extract(stego, extracted_data, data.size(), lsb_opt);

		/*
		 * write the first cover and stego images
		 */
		if(n < first){
			std::stringstream fcover;
			std::stringstream fstego;
			fcover << "lsb_cover_" << n << ".ppm";
			fstego << "lsb_stego_" << n << ".ppm";
			n++;
			cv::imwrite(fcover.str(), cover);
			cv::imwrite(fstego.str(), stego);
		}

		/*
		 * test the extraction
		 */
		if(data != extracted_data){
			std::cout << "Embeded (" << data.size() << "): " << std::endl;
			print_data(data);
			std::cout << "Extracted (" << data.size() << "): " << std::endl;
			print_data(extracted_data);
			std::cerr << "Extracted data is different from embedded data!"
				  << std::endl;
			cv::imwrite("lsb_cover_error.ppm", cover);
			cv::imwrite("lsb_stego_error.ppm", stego);
			exit(EXIT_FAILURE);
		}

	}
}

int main()
{
	srand(time(NULL));

	std::string cover_image_path(COVER_IMAGE_PATH);

	test_grayscale(glob(cover_image_path + "/*.pgm"));
	test_color(glob(cover_image_path + "/*.ppm"));

	return 0;
}
