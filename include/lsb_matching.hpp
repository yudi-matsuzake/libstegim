#pragma once

#include <vector>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace stegim {

/** Embeds the `data` in `cover` image using the `key` 
  * and writes the result in `stego`. This is a implementation
  * of lsb matching algorithm, inspired by the white papers
  * "An Implementation of Key-Based Digital Signal Steganography"
  * and "LSB Matching Revised".
  *
  * @param cover	The cover image. Must be CV_8UC{1,3,4} type.
  * @param stego	The stego image buffer. Must be CV_8UC{1,3,4} type.
  * @param data		Data to be embedded in `cover`
  * @param key		The key to be used in the embedding process. To the
  * 			operation be reversible, the same key must to be used
  * 			in the extraction process.
  *
  * @return		The number of bytes successfully embedded.
  */
void lsb_matching_embed(
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const std::vector<char>& key);

void lsb_matching_embed(
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const std::string key);

/** Extracts the embedded data from `stego` usign `key`
  * and writes the result in `data` vector. Extraction is
  * realized taking into consideration the embedding process
  * was `lsb_matching_embed`.
  *
  * @param stego	The stego image buffer. Must be CV_8UC{1,3,4} type.
  * @param data		Vector buffer to be write with embedded data from `stego`
  * @param size		The size of the embedded data in `stego`
  * @param key		The key to be used in the embedding process.
  *
  * @see lsb_matching_embed
  *
  */
void lsb_matching_extract(
	const cv::Mat& stego,
	std::vector<char>& data,
	size_t size,
	const std::vector<char>& key);

void lsb_matching_extract(
	const cv::Mat& stego,
	std::vector<char>& data,
	size_t size,
	const std::string& key);

/*
 * end of stegim namespace
 */
}
