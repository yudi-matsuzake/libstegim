#include <cstdint>

#include "lsb_matching.hpp"

#define LSB(X) ((X)&1)

#define BIGGEST_64BIT_PRIME 18446744073709551557ULL

typedef std::pair<cv::Point2i, cv::Point2i> lsbm_pair;

/*
 * simple hash function to derivate a key to a uint64_t
 */
inline uint64_t prime_hash(const char* str, size_t len)
{
	uint64_t h = BIGGEST_64BIT_PRIME;

	size_t i = 0;
	for(i = 0; i < len; i++)
		h = h*31 + str[i] + 1;

	return h;
}

/*
 * this functions is the same of the paper
 * "LSB Matching Revised"
 */
inline uchar correlation_function(uchar a, uchar b)
{
	return LSB(a/2 + b);
}

/*
 * return +/- 1 if the pixel is not saturated
 */
int rand_plus_minus_one(uchar c)
{
	static std::random_device random;

	if(c == 0)
		return 1;
	else if(c == UCHAR_MAX)
		return -1;

	return (random()%2)?1:-1;
}

/*
 * embed the `ibit`-th and the `ibit`+1-th bits of
 * `m` in `s0` and `s1`, respectively
 */
inline void lsbm_embed_pixel_little_endian(
		uchar m,
		uchar ibit,
		const uchar c0,
		const uchar c1,
		uchar& s0,
		uchar& s1)
{
	int m0 = LSB(m >> ibit);
	int m1 = LSB(m >> (ibit+1));

	if(m0 == LSB(c0)){
		if(m1 == correlation_function(c0, c1))
			s1 = c1;
		else
			s1 = c1 + rand_plus_minus_one(c1);

		s0 = c0;
	}else{

		/*
		 * these 2 next conditions are added to the algorithm
		 * to treat the saturated pixel issue
		 */
		if(c0 == 0){
			if(m1 == correlation_function(c0 + 1, c1))
				s0 = c0 + 1;
			else
				s0 = c0 + 3;
		}else if(c0 == 255){
			if(m1 == correlation_function(c0 - 1, c1))
				s0 = c0 - 1;
			else
				s0 = c0 - 3;
		}else if(m1 == correlation_function(c0 - 1, c1)){
			s0 = c0 - 1;
		}else{
			s0 = c0 + 1;
		}

		s1 = c1;
	}

}

/*
 * extract the embed message in `s0` and `s1`
 * and put in `ibit`-th bit, and `ibit`+1-th bit
 * of d respectively
 */
inline uchar lsbm_extract_pixel_little_endian(
		uchar d,
		uchar s0,
		uchar s1,
		uchar ibit)
{
	int m0 = LSB(s0);
	int m1 = correlation_function(s0, s1);

	d = (d & ~(1 << ibit)) | m0 << ibit;
	ibit++;
	d = (d & ~(1 << ibit)) | m1 << ibit;

	return d;
}


/*
 * returns a list of pair of points shuffled
 */
std::vector<lsbm_pair> lsbm_pair_shuffled(
	int rows,
	int cols,
	std::default_random_engine& random_generator,
	std::uniform_int_distribution<int>& uniform)
{
	std::vector<cv::Point2i> point;
	std::vector<lsbm_pair> v;

	int n_pixel = rows*cols;
	for(int i=0; i<n_pixel; i++)
		point.push_back(cv::Point2i(i/cols, i%cols));

	if(n_pixel%2 != 0){
		v[n_pixel - 1].second.x = -1;
		v[n_pixel - 1].second.y = -1;
	}

	/*
	 * shuffle!
	 */
	for(size_t i=0; i<point.size(); i++){
		int random_index = uniform(random_generator)%point.size();
		std::swap(point[i], point[random_index]);
	}

	for(size_t i=0; i<point.size(); i += 2){
		v.push_back(lsbm_pair(point[i], point[i+1]));
	}

	return v;
}

void lsb_maching_embed_single_channel(
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const std::vector<char>& key)
{
	int64_t hash_seed = prime_hash(key.data(), key.size());

	/*
	 * default random generator
	 */
	std::default_random_engine random_generator(hash_seed);

	/*
	 * default random distribution
	 */
	std::uniform_int_distribution<int> uniform;

	std::vector<lsbm_pair> pair = lsbm_pair_shuffled(
		cover.rows,
		cover.cols,
		random_generator,
		uniform);

	size_t n_bytes = 0;
	size_t i = 0;
	size_t n_bits = 0;
	while(i < pair.size()){
		lsbm_pair& p = pair[i];

		const uchar* ptr_first_cover =
			cover.ptr<uchar>(p.first.x) + p.first.y;

		const uchar* ptr_second_cover =
			cover.ptr<uchar>(p.second.x) + p.second.y;

		uchar* ptr_first_stego = 
			stego.ptr<uchar>(p.first.x) + p.first.y;

		uchar* ptr_second_stego =
			stego.ptr<uchar>(p.second.x) + p.second.y;

		if(n_bytes < data.size()){

			lsbm_embed_pixel_little_endian(
					data[n_bytes],
					n_bits%CHAR_BIT,
					*ptr_first_cover,
					*ptr_second_cover,
					*ptr_first_stego,
					*ptr_second_stego);

			n_bits += 2;
			n_bytes = n_bits/CHAR_BIT;
		}else{
			*ptr_first_stego = *ptr_first_cover;
			*ptr_second_stego = *ptr_second_cover;
		}

		i++;
	}
}

void lsb_matching_extract_single_channel(
	const cv::Mat& stego,
	std::vector<char>& data,
	size_t size,
	const std::vector<char>& key)
{
	int64_t hash_seed = prime_hash(key.data(), key.size());

	/*
	 * default random generator
	 */
	std::default_random_engine random_generator(hash_seed);

	/*
	 * default random distribution
	 */
	std::uniform_int_distribution<int> uniform;

	std::vector<lsbm_pair> pair = lsbm_pair_shuffled(
		stego.rows,
		stego.cols,
		random_generator,
		uniform);

	size_t i = 0;
	size_t n_bytes = 0;
	size_t n_bits = 0;
	while(i < pair.size() && n_bytes < size){

		lsbm_pair& p = pair[i];
		const uchar* ptr_first_stego =
			stego.ptr<uchar>(p.first.x) + p.first.y;
		const uchar* ptr_second_stego =
			stego.ptr<uchar>(p.second.x) + p.second.y;

		data[n_bytes] = lsbm_extract_pixel_little_endian(
				data[n_bytes],
				*ptr_first_stego,
				*ptr_second_stego,
				n_bits%CHAR_BIT);

		n_bits += 2;
		n_bytes = n_bits/CHAR_BIT;

		i++;
	}

}

void stegim::lsb_matching_embed(
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const std::vector<char>& key)
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
		lsb_maching_embed_single_channel(cover, stego, data, key);
	}else{
		assert(0);
	}
}

void stegim::lsb_matching_embed(
	const cv::Mat& cover,
	cv::Mat& stego,
	const std::vector<char>& data,
	const std::string key)
{
	std::vector<char> k(key.data(), key.data() + key.size());
	return stegim::lsb_matching_embed(cover, stego, data, k);
}

void stegim::lsb_matching_extract(
	const cv::Mat& stego,
	std::vector<char>& data,
	size_t size,
	const std::vector<char>& key)
{

	assert(	stego.type() == CV_8UC1 ||
		stego.type() == CV_8UC3 ||
		stego.type() == CV_8UC4);
	assert(stego.cols && stego.rows);

	data.clear();
	data.resize(size);

	/*
	 * Separate the lsb_extract in single_channel and multiple_channel
	 * just for optimization in the comparison number for single channel.
	 * Maybe there is a more elegant way to do this, with the same result.
	 */
	if(stego.type() == CV_8UC1){
		lsb_matching_extract_single_channel(stego, data, size, key);
	}else{
		assert(0);
	}
}

void stegim::lsb_matching_extract(
	const cv::Mat& stego,
	std::vector<char>& data,
	size_t size,
	const std::string& key)
{
	std::vector<char> k(key.data(), key.data() + key.size());
	lsb_matching_extract(stego, data, size, k);
}
