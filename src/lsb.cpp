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

	return (pixel & ~1) | x;
}

/*
 * extracts the last bit in `pixel` and put in `ibit` bit position in `data`
 * and returns it
 */
uchar lsb_extract_pixel_little_endian(uchar pixel, uchar data, uchar ibit)
{
	assert(ibit < CHAR_BIT);

	int last_bit = pixel & 1;

	return (data & ~(1 << ibit)) | (last_bit << ibit);
}

/*
 * embeds the `data` in `src` in a sigle channel
 */
void lsb_embed_single_channel(
	const cv::Mat& src,
	cv::Mat& dst,
	const std::vector<char>& data,
	int offset)
{
	size_t rows = src.rows;
	size_t cols = src.cols;

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = offset/src.cols;
	size_t j_ini = offset%src.cols;

	if(src.isContinuous())
	{
		cols *= rows;
		rows = 1;
		i_ini = 0;
		j_ini = offset;
	}

	for(	size_t i = i_ini, n_bytes = 0, n_bits = 0;
		i < rows && n_bytes < data.size();
		i++){

		const uchar* ptr_src = src.ptr<uchar>(i);
		uchar* ptr_dst = dst.ptr<uchar>(i);
		for(	size_t j = j_ini;
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
 * embeds the data in a multichannel image.
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

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = offset/src.cols;
	size_t j_ini = offset%src.cols;

	if(src.isContinuous())
	{
		cols *= rows;
		rows = 1;
		i_ini = 0;
		j_ini = offset;
	}

	/*
	 * vector for a channel i is for embed
	 */
	std::vector<bool> embed_channel;
	embed_channel.push_back(B);
	embed_channel.push_back(G);
	embed_channel.push_back(R);
	embed_channel.push_back(A);

	for(	size_t i = i_ini, n_bytes = 0, n_bits = 0;
		i < rows && n_bytes < data.size();
		i++){

		const uchar* ptr_src = src.ptr<uchar>(i);
		uchar* ptr_dst = dst.ptr<uchar>(i);
		for(	size_t j = j_ini;
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

/*
 * extracts the data embedded in `stego` in one channel
 * beginning in `offset`-th pixel and writes
 * it in `data` vector
 */
void lsb_extract_single_channel(
	const cv::Mat& stego,
	std::vector<char>& data,
	int offset,
	int size)
{

	size_t rows = stego.rows;
	size_t cols = stego.cols;

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = offset/stego.cols;
	size_t j_ini = offset%stego.cols;

	if(stego.isContinuous())
	{
		cols *= rows;
		rows = 1;
		i_ini = 0;
		j_ini = offset;
	}

	size_t max_bytes;
	if(size == -1)
		max_bytes = (stego.cols*stego.rows)/CHAR_BIT;
	else
		max_bytes = size;

	data.resize(max_bytes, 0);

	for(	size_t i = i_ini, n_bytes = 0, n_bits = 0;
		i < rows && n_bytes < max_bytes;
		i++){

		const uchar* ptr_stego = stego.ptr<uchar>(i);
		for(	size_t j = j_ini;
			j < cols && n_bytes < max_bytes;
			j++){

			/*
			 * extracting
			 */
			data[n_bytes] = lsb_extract_pixel_little_endian(
						*ptr_stego,
						data[n_bytes],
						n_bits%CHAR_BIT);

			n_bits++;
			n_bytes = n_bits/CHAR_BIT;
			ptr_stego += stego.channels();
		}
	}
}

/*
 * extracts the data embedded in `stego` in multiple channels
 * beginning in `offset`-th pixel and writes
 * it in `data` vector
 */
void lsb_extract_multiple_channel(
	const cv::Mat& stego,
	std::vector<char>& data,
	int offset,
	int size,
	bool B,
	bool G,
	bool R,
	bool A)
{
	size_t rows = stego.rows;
	size_t cols = stego.cols;

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = offset/stego.cols;
	size_t j_ini = offset%stego.cols;

	if(stego.isContinuous())
	{
		cols *= rows;
		rows = 1;
		i_ini = 0;
		j_ini = offset;
	}

	size_t max_bytes;
	if(size == -1)
		max_bytes = (stego.cols*stego.rows)/CHAR_BIT;
	else
		max_bytes = size;

	data.resize(max_bytes, 0);

	/*
	 * vector for a channel i is for embed
	 */
	std::vector<bool> embed_channel;
	embed_channel.push_back(B);
	embed_channel.push_back(G);
	embed_channel.push_back(R);
	embed_channel.push_back(A);

	for(	size_t i = i_ini, n_bytes = 0, n_bits = 0;
		i < rows && n_bytes < max_bytes;
		i++){

		const uchar* ptr_stego = stego.ptr<uchar>(i);
		for(	size_t j = j_ini;
			j < cols && n_bytes < max_bytes;
			j++){

			for(int c = 0; c < stego.channels(); c++){

				/*
				 * if this channel has embedded data
				 */
				if(embed_channel[c]){
					/*
					 * extracting
					 */
					data[n_bytes] = lsb_extract_pixel_little_endian(
								*ptr_stego,
								data[n_bytes],
								n_bits%CHAR_BIT);

					n_bits++;
					n_bytes = n_bits/CHAR_BIT;
					ptr_stego++;
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
	assert(src.cols && src.rows);
	src.copyTo(dst);

	/*
	 * Separate the lsb_embed in single_channel and multiple_channel
	 * just for optimization in the comparison number for single channel.
	 * Maybe there is a more elegant way to do this, with the same result.
	 */
	if(src.type() == CV_8UC1){
		lsb_embed_single_channel(src, dst, data, offset);
	}else{
		assert(B + G + R + A);
		lsb_embed_multiple_channel(src, dst, data, offset, B, G, R, A);
	}
}

void stegim::lsb_extract(
	const cv::Mat& stego,
	std::vector<char>& data,
	int offset,
	int size,
	bool B,
	bool G,
	bool R,
	bool A)
{
	data.clear();

	assert(	stego.type() == CV_8UC1 ||
		stego.type() == CV_8UC3 ||
		stego.type() == CV_8UC4);
	assert(stego.cols && stego.rows);

	/*
	 * Separate the lsb_extract in single_channel and multiple_channel
	 * just for optimization in the comparison number for single channel.
	 * Maybe there is a more elegant way to do this, with the same result.
	 */
	if(stego.type() == CV_8UC1){
		lsb_extract_single_channel(stego, data, offset, size);
	}else{
		assert(B + G + R + A);
		lsb_extract_multiple_channel(stego, data, offset, size, B, G, R, A);
	}
}
