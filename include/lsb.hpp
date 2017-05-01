#pragma once

#include <vector>
#include <random>

#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace stegim {

/** The `lsb_options` class functions is to provide a
  * variable optional arguments facility.
  */
class lsb_options {
public:
	/** `lsb_options` constructor have default arguments
	  * but this class also provides set functions for
	  * convenience.
	  *
	  * @param b		whether the B (blue) channel will be used for
	  *			embedding. In case of the `cover` image is
	  *			grayscale, this argument will be ignored.
	  * @param g		whether the G (green) channel will be used for
	  *			embedding. In case of the `cover` image is
	  *			grayscale, this argument will be ignored.
	  * @param r		whether the R (red) channel will be used for
	  *			embedding. In case of the `cover` image is
	  *			grayscale, this argument will be ignored.
	  * @param a		whether the A (alpha) channel will be used for
	  *			embedding. In case of the `cover` image is
	  *			does not have alpha channel, this argument
	  *			will be ignored.
	  * @param offset	The pixel offset to be the begin of the
	  *			embedding process.
	  */
	lsb_options(
		bool b = true,
		bool g = true,
		bool r = true,
		bool a = false,
		int offset = 0);

	virtual ~lsb_options();

	virtual lsb_options& set_b(bool b);
	virtual lsb_options& set_g(bool g);
	virtual lsb_options& set_r(bool r);
	virtual lsb_options& set_a(bool r);
	virtual lsb_options& set_offset(int offset);

	virtual bool get_b() const;
	virtual bool get_g() const;
	virtual bool get_r() const;
	virtual bool get_a() const;
	virtual int get_offset() const;

private:
	bool b, g, r, a;
	int offset;
};

/** Perform a naive lsb algorithm.
  * TODO: A more detailed description of lsb.
  *
  * @param cover	The cover image. Must be CV_8UC{1,3,4} type.
  * @param stego	The stego image buffer. Must be CV_8UC{1,3,4} type.
  * @param data		Vector for embedding the data
  * @param lsb_opt	Optional arguments of lsb_embed
  *
  * @see lsb_options
  */
void lsb_embed (
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const lsb_options& lsb_opt = lsb_options());

/** Extract the embedded data from `stego` and write it on `data`
  * vector.
  * TODO: A more detailed description of lsb_extract.
  *
  * @param stego	Image containing the embed data.
  * @param data		Vector to return the data on
  * @param size		The size of the message embedded in bytes
  * @param lsb_opt	Optional arguments of lsb_extract
  */
void lsb_extract (
	const cv::Mat& stego,
	std::vector<char>& data,
	int size = -1,
	const lsb_options& lsb_opt = lsb_options());

/*
 * end of stegim namespace
 */
}
