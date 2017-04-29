#include "lsb.hpp"
#include <climits>

/*
 * embeds the `bit` in `data` positioned in `ibit` bit index
 * and returns it
 */
uchar lsb_embed_pixel_little_endian(uchar pixel, uchar data, uchar ibit)
{
	assert(ibit < CHAR_BIT);

	/*
	 * set the the last bit of pixel
	 * with `x`
	 */
	int x = (data >> ibit) & 1;

	return pixel & ( ~1 | x );
}

void lsb_embed_single_channel(
	const cv::Mat& src,
	cv::Mat& dst,
	const std::vector<char>& data,
	int offset)
{
	size_t rows = src.rows;
	size_t cols = src.cols;

	if(src.isContinuous())
	{
		cols *= rows;
		rows = 1;
	}

	for(	size_t i = 0, n_bytes = 0, n_bits = 0;
		i < rows && n_bytes < data.size();
		i++){

		const uchar* ptr_src = src.ptr<uchar>(i);
		uchar* ptr_dst = dst.ptr<uchar>(i);
		for(	size_t j = 0;
			j < cols && n_bytes < data.size();
			j++){

			/*
			 * embedding
			 */
			*ptr_dst = lsb_embed_pixel_little_endian(
					*ptr_src,
					data[n_bytes],
					n_bits%CHAR_BIT);

			n_bits++;
			n_bytes = n_bits/CHAR_BIT;
			ptr_src += src.channels();
			ptr_dst += dst.channels();
		}
	}
}

/*
 * embeds the data in a multichannel image
 */
void lsb_embed_multiple_channel(
	const cv::Mat& src,
	cv::Mat& dst,
	const std::vector<char>& data,
	int offset,
	bool B,
	bool G,
	bool R,
	bool A)
{
	size_t rows = src.rows;
	size_t cols = src.cols;

	if(src.isContinuous())
	{
		cols *= rows;
		rows = 1;
	}

	/*
	 * vector for a channel i is for embed
	 */
	std::vector<bool> embed_channel;
	embed_channel.push_back(B);
	embed_channel.push_back(G);
	embed_channel.push_back(R);
	embed_channel.push_back(A);

	for(	size_t i = 0, n_bytes = 0, n_bits = 0;
		i < rows && n_bytes < data.size();
		i++){

		const uchar* ptr_src = src.ptr<uchar>(i);
		uchar* ptr_dst = dst.ptr<uchar>(i);
		for(	size_t j = 0;
			j < cols && n_bytes < data.size();
			j++){

			for(int c = 0; c < src.channels(); c++){

				/*
				 * if embeds in this channel
				 */
				if(embed_channel[c]){

					/*
					 * embedding
					 */
					*ptr_dst = lsb_embed_pixel_little_endian(
							*ptr_src,
							data[n_bytes],
							n_bits%CHAR_BIT);

					n_bits++;
					n_bytes = n_bits/CHAR_BIT;

					ptr_src++;
					ptr_dst++;
				}
			}
		}
	}
}

void stegim::lsb_embed (
	const cv::Mat& src,
	cv::Mat& dst,
	const std::vector<char>& data,
	int offset,
	bool B,
	bool G,
	bool R,
	bool A)
{
	assert(	src.type() == CV_8UC1 ||
		src.type() == CV_8UC3 ||
		src.type() == CV_8UC4);
	src.copyTo(dst);

	if(src.type() == CV_8UC1){
		lsb_embed_single_channel(src, dst, data, offset);
	}else{
		assert(B + G + R + A);
		lsb_embed_multiple_channel(src, dst, data, offset, B, G, R, A);
	}
}
