#include "lsb.hpp"

/*  _     _        __                  _   _                 
   | |___| |__    / _|_   _ _ __   ___| |_(_) ___  _ __  ___ 
   | / __| '_ \  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
   | \__ \ |_) | |  _| |_| | | | | (__| |_| | (_) | | | \__ \
   |_|___/_.__/  |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
                                                             */


/*
 * embeds the `bit` in `data` positioned in `ibit` bit index
 * and returns it
 */
inline uchar lsb_embed_pixel_little_endian(uchar pixel, uchar data, uchar ibit)
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
inline uchar lsb_extract_pixel_little_endian(uchar pixel, uchar data, uchar ibit)
{
	assert(ibit < CHAR_BIT);

	int last_bit = pixel & 1;

	return (data & ~(1 << ibit)) | (last_bit << ibit);
}

/*
 * copy image beginning in `begin` and ending in `end`
 * range of pixels
 */
void copy_mat_range(
	cv::Mat& dst,
	const cv::Mat& src,
	int begin = 0,
	int end = 0)
{

	assert(begin < src.cols*src.rows*src.channels());
	assert(end <= src.cols*src.rows*src.channels());

	if(begin >= end)
		end = src.cols*src.rows*src.channels();

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = begin/src.cols;
	size_t j_ini = begin%src.cols;

	int rows = src.rows;
	int cols = src.cols;
	 
	if(src.isContinuous())
	{
		cols *= rows;
		rows = 1;
		i_ini = 0;
		j_ini = begin;
	}

	int n_uc_to_copy = end - begin;

	for(int i = i_ini, uc_count = 0; i < rows && uc_count < n_uc_to_copy; i++){
		const uchar* ptr_src = src.ptr<uchar>(i);
		uchar* ptr_dst = dst.ptr<uchar>(i);

		for (int j = j_ini; j < cols && uc_count < n_uc_to_copy; j++){
			/*
			 * reset j_ini
			 */
			if(j_ini > 0){
				ptr_src += j_ini*src.channels();
				ptr_dst += j_ini*dst.channels();
				j_ini = 0;
			}

			for(	int c = 0;
				c < src.channels();
				c++){

				*ptr_dst = *ptr_src;

				ptr_dst++;
				ptr_src++;
			}
			uc_count++;
		}
	}
}

/*
 * embeds the `data` in `cover` in a sigle channel
 */
void lsb_embed_single_channel(
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const stegim::lsb_options& lsb_opt)
{
	size_t rows = cover.rows;
	size_t cols = cover.cols;
	int offset = lsb_opt.get_offset();

	if(offset)
		copy_mat_range(stego, cover, 0, offset);

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = offset/cover.cols;
	size_t j_ini = offset%cover.cols;

	if(cover.isContinuous()){
		cols *= rows;
		rows = 1;
		i_ini = 0;
		j_ini = offset;
	}

	size_t n_bits = 0;
	for(	size_t i = i_ini, n_bytes = 0;
		i < rows && n_bytes < data.size();
		i++){

		const uchar* ptr_cover = cover.ptr<uchar>(i);
		uchar* ptr_stego = stego.ptr<uchar>(i);
		for(	size_t j = j_ini;
			j < cols && n_bytes < data.size();
			j++){

			/*
			 * reset j_ini
			 */
			if(j_ini > 0){
				ptr_cover += j_ini;
				ptr_stego += j_ini;
				j_ini = 0;
			}

			/*
			 * embedding
			 */
			*ptr_stego = lsb_embed_pixel_little_endian(
					*ptr_cover,
					data[n_bytes],
					n_bits%CHAR_BIT);

			n_bits++;
			n_bytes = n_bits/CHAR_BIT;
			ptr_cover += cover.channels();
			ptr_stego += stego.channels();

		}
	}

	/*
	 * we can use the offset+n_bits count to the rest of the image
	 */
	copy_mat_range(stego, cover, offset+n_bits);
}

/*
 * embeds the data in a multichannel image.
 */
void lsb_embed_multiple_channel(
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const stegim::lsb_options& lsb_opt)
{
	size_t rows = cover.rows;
	size_t cols = cover.cols;

	int offset = lsb_opt.get_offset();

	if(offset)
		copy_mat_range(stego, cover, 0, offset);

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = offset/cover.cols;
	size_t j_ini = offset%cover.cols;

	if(cover.isContinuous()){
		cols *= rows;
		rows = 1;
		i_ini = 0;
		j_ini = offset;
	}

	/*
	 * vector for a channel i is for embed
	 */
	std::vector<bool> embed_channel;
	embed_channel.push_back(lsb_opt.get_b());
	embed_channel.push_back(lsb_opt.get_g());
	embed_channel.push_back(lsb_opt.get_r());
	embed_channel.push_back(lsb_opt.get_a());

	size_t n_pixel = 0;
	for(	size_t i = i_ini, n_bytes = 0, n_bits = 0;
		i < rows && n_bytes < data.size();
		i++){

		const uchar* ptr_cover = cover.ptr<uchar>(i);
		uchar* ptr_stego = stego.ptr<uchar>(i);
		for(	size_t j = j_ini;
			j < cols && n_bytes < data.size();
	 		j++){

			/*
			 * reset j_ini
			 */
			if(j_ini > 0){
				ptr_cover += j_ini*cover.channels();
				ptr_stego += j_ini*stego.channels();
				j_ini = 0;
			}

			for(int c = 0; c < cover.channels() && n_bytes < data.size(); c++){

				/*
				 * if embeds in this channel
				 */
				if(embed_channel[c]){

					/*
					 * embedding
					 */
					ptr_stego[c] = lsb_embed_pixel_little_endian(
							ptr_cover[c],
							data[n_bytes],
							n_bits%CHAR_BIT);

					n_bits++;
					n_bytes = n_bits/CHAR_BIT;

				}else{
					ptr_stego[c] = ptr_cover[c];
				}
			}

			n_pixel++;
			ptr_cover += cover.channels();
			ptr_stego += stego.channels();
		}
	}

	/*
	 * we can use the offset+n_bits count to the rest of the image
	 */
	copy_mat_range(stego, cover, offset+n_pixel);
}

/*
 * extracts the data embedded in `stego` in one channel
 * beginning in `offset`-th pixel and writes
 * it in `data` vector
 */
void lsb_extract_single_channel(
	const cv::Mat& stego,
	std::vector<char>& data,
	int size,
	const stegim::lsb_options& lsb_opt)
{

	size_t rows = stego.rows;
	size_t cols = stego.cols;

	int offset = lsb_opt.get_offset();

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = offset/stego.cols;
	size_t j_ini = offset%stego.cols;

	if(stego.isContinuous()){
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

	for(	size_t i = i_ini, n_bits = 0, n_bytes = 0;
		i < rows && n_bytes < max_bytes;
		i++){

		const uchar* ptr_stego = stego.ptr<uchar>(i);
		for(	size_t j = j_ini;
			j < cols && n_bytes < max_bytes;
			j++){

			/*
			 * reset j_ini
			 */
			if(j_ini > 0){
				ptr_stego += j_ini;
				j_ini = 0;
			}

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
	int size,
	const stegim::lsb_options& lsb_opt)
{
	size_t rows = stego.rows;
	size_t cols = stego.cols;

	int offset = lsb_opt.get_offset();

	/*
	 * calculate the i, j inicial position
	 */
	size_t i_ini = offset/stego.cols;
	size_t j_ini = offset%stego.cols;

	if(stego.isContinuous()){
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
	embed_channel.push_back(lsb_opt.get_b());
	embed_channel.push_back(lsb_opt.get_g());
	embed_channel.push_back(lsb_opt.get_r());
	embed_channel.push_back(lsb_opt.get_a());

	for(	size_t i = i_ini, n_bytes = 0, n_bits = 0;
		i < rows && n_bytes < max_bytes;
		i++){

		const uchar* ptr_stego = stego.ptr<uchar>(i);
		for(	size_t j = j_ini;
			j < cols && n_bytes < max_bytes;
			j++){

			/*
			 * reset j_ini
			 */
			if(j_ini > 0){
				ptr_stego += j_ini*stego.channels();
				j_ini = 0;
			}

			for(int c = 0; c < stego.channels(); c++){

				/*
				 * if this channel has embedded data
				 */
				if(embed_channel[c]){
					/*
					 * extracting
					 */
					data[n_bytes] = lsb_extract_pixel_little_endian(
								ptr_stego[c],
								data[n_bytes],
								n_bits%CHAR_BIT);

					n_bits++;
					n_bytes = n_bits/CHAR_BIT;
				}
			}

			ptr_stego += stego.channels();
		}
	}
}

void stegim::lsb_embed (
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const stegim::lsb_options& lsb_opt)
{
	assert(	cover.type() == CV_8UC1 ||
		cover.type() == CV_8UC3 ||
		cover.type() == CV_8UC4);
	assert(cover.cols && cover.rows);
	stego.create(cover.size(), cover.type());

	/*
	 * Separation of the lsb_embed in single_channel and multiple_channel
	 * is just for optimization in the comparison number for single channel.
	 * Maybe there is a more elegant way to do this, with the same result.
	 */
	if(cover.type() == CV_8UC1){
		lsb_embed_single_channel(cover, stego, data, lsb_opt);
	}else{
		assert(lsb_opt.get_b()
			|| lsb_opt.get_g()
			|| lsb_opt.get_r()
			|| lsb_opt.get_a());

		lsb_embed_multiple_channel(cover, stego, data, lsb_opt);
	}
}

void stegim::lsb_extract(
	const cv::Mat& stego,
	std::vector<char>& data,
	int size,
	const stegim::lsb_options& lsb_opt)
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
		lsb_extract_single_channel(stego, data, size, lsb_opt);
	}else{
		assert(lsb_opt.get_b()
			|| lsb_opt.get_g()
			|| lsb_opt.get_r()
			|| lsb_opt.get_a());
		lsb_extract_multiple_channel(stego, data, size, lsb_opt);
	}
}

/*  _     _                   _   _                 
   | |___| |__     ___  _ __ | |_(_) ___  _ __  ___ 
   | / __| '_ \   / _ \| '_ \| __| |/ _ \| '_ \/ __|
   | \__ \ |_) | | (_) | |_) | |_| | (_) | | | \__ \
   |_|___/_.__/   \___/| .__/ \__|_|\___/|_| |_|___/
                       |_|                          */

/*
 * lsb_options
 */
stegim::lsb_options::lsb_options(
	bool b,
	bool g,
	bool r,
	bool a,
	int offset)
	: b(b),
	g(g),
	r(r),
	a(a),
	offset(offset)
{}

stegim::lsb_options::~lsb_options()
{};

stegim::lsb_options& stegim::lsb_options::set_b(bool b)
{
	this->b = b;
	return *this;
}

stegim::lsb_options& stegim::lsb_options::set_g(bool g)
{
	this->g = g;
	return *this;
}

stegim::lsb_options& stegim::lsb_options::set_r(bool r)
{
	this->r = r;
	return *this;
}

stegim::lsb_options& stegim::lsb_options::set_a(bool a)
{
	this->a = a;
	return *this;
}

stegim::lsb_options& stegim::lsb_options::set_offset(int offset)
{
	this->offset = offset;
	return *this;
}

bool stegim::lsb_options::get_b() const
{
	return this->b;
}

bool stegim::lsb_options::get_g() const
{
	return this->g;
}

bool stegim::lsb_options::get_r() const
{
	return this->r;
}

bool stegim::lsb_options::get_a() const
{
	return this->a;
}

int stegim::lsb_options::get_offset() const
{
	return this->offset;
}
