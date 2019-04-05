// Copyright © 2019 by Spectrico
// Licensed under the MIT License

#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

template <typename T>
std::vector<size_t> SortIndexes(const std::vector<T> &v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });

	return idx;
}

std::vector<std::string> readClassNames(std::string filename)
{
	std::vector<std::string> classNames;

	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}

	fp.close();
	return classNames;
}

int main(int argc, char** argv)
{
	cv::dnn::Net net;
	const std::string modelFile = "model-weights-spectrico-car-colors-mobilenet-224x224-052EAC82.pb";
	//! [Initialize network]
	net = cv::dnn::readNetFromTensorflow(modelFile);
	if (net.empty())
	{
		std::cerr << "Can't load network by using the model file: " << std::endl;
		std::cerr << modelFile << std::endl;
		exit(-1);
	}

	std::vector<std::string> classNames = readClassNames("labels.txt");

	std::string imageFile = argc == 2 ? argv[1] : "car.jpg";
	//! [Prepare blob]
	cv::Mat img = cv::imread(imageFile, cv::IMREAD_COLOR);
	if (img.empty() || !img.data)
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	int center_crop_size = (img.cols <= img.rows) ? img.cols : img.rows;
	img = img(cv::Rect((img.cols - center_crop_size) / 2, (img.rows - center_crop_size) / 2, center_crop_size, center_crop_size));
	cv::resize(img, img, cv::Size(224, 224));

	cv::Mat inputBlob = cv::dnn::blobFromImage(img, 0.0078431372549019607843137254902, cv::Size(224, 224), cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);   //Convert Mat to image batch
	std::string inBlobName = "input_1";
	//! [Set input blob]
	net.setInput(inputBlob, inBlobName);        //set the network input

	std::string outBlobName = "softmax/Softmax";

	cv::TickMeter tm;
	tm.start();

	cv::Mat result;
	//! [Make forward pass]
	result = net.forward(outBlobName);       //compute output

	tm.stop();

	std::cout << "Inference time, ms: " << tm.getTimeMilli() << std::endl;
	std::cout << "Probabilities: " << std::endl;

	cv::Mat probMat = result.reshape(1, 1);
	std::vector<float>vec(probMat.begin<float>(), probMat.end<float>());
	int top = 0;
	for (auto i : SortIndexes(vec)) {
		std::cout << classNames.at(i) << ":\t" << vec[i]*100 << " %" << std::endl;
		if (++top == 3)
			break;
	}

	return 0;
}
