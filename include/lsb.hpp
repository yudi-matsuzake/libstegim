#pragma once

#include <vector>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace stegim {

/** Perform a naive lsb algorithm.
  * TODO: A more detailed description of lsb.
  *
  * @param src		The cover image. Must be CV_8UC{1,3,4} type.
  * @param dst		The stego image buffer. Must be CV_8UC{1,3,4} type.
  * @param data		Vector for embedding the data
  * @param offset	The pixel offset to begin the embedding
  * @param B		whether the B channel will be used for embedding.
  * 			In case of the `src` image is grayscale, this argument
  * 			will be ignored.
  * @param G		whether the G channel will be used for embedding.
  * 			In case of the `src` image is grayscale, this argument
  * 			will be ignored.
  * @param R		whether the R channel will be used for embedding.
  * 			In case of the `src` image is grayscale, this argument
  * 			will be ignored.
  * @param A		whether the A channel will be used for embedding.
  * 			In case of the `src` image is does not have alpha channel,
  * 			this argument will be ignored.
  */
void lsb_embed (
	const cv::Mat& src,
	cv::Mat& dst,
	const std::vector<char>& data,
	int offset = 0,
	bool B = true,
	bool G = true,
	bool R = true,
	bool A = true);

/** Extract the embedded data from `stego` and write it on `data`
  * vector.
  * TODO: A more detailed description of lsb_extract.
  *
  * @param stego	Image containing the embed data.
  * @param data		Vector to return the data on
  * @param offset	The pixel offset to begin the embedding
  * @param size		The size of the message embedded in bytes
  * @param B		whether the B channel was used for embedding.
  * 			In case of the `stego` image is grayscale, this argument
  * 			will be ignored.
  * @param G		whether the G channel was used for embedding.
  * 			In case of the `stego` image is grayscale, this argument
  * 			will be ignored.
  * @param R		whether the R channel was used for embedding.
  * 			In case of the `stego` image is grayscale, this argument
  * 			will be ignored.
  * @param A		whether the A channel was used for embedding.
  * 			In case of the `stego` image does not have an alpha channel,
  * 			this argument will be ignored.
  */
void lsb_extract (
	const cv::Mat& stego,
	std::vector<char>& data,
	int offset = 0,
	int size = -1,
	bool B = true,
	bool G = true,
	bool R = true,
	bool A = true);

/*
 * end of stegim namespace
 */
}
