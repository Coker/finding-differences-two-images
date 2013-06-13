
#include <iostream>
#include <cmath>

#include <opencv\cv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;

#define MINIMUM_GRAD_MAGNITUDE_FOR_ORIENTATION 0.05
#define PI 3.14159265

void getHOGFeatures1(Mat InputImage, Mat & Histogram);
bool findHistogramDif(cv::Mat hist1, cv::Mat hist2);
void getDiff(void);

int main()
{
	cv::Mat histogram1,
			histogram2;

	histogram1.create(1, 180, DataType<int>::type);
	histogram2.create(1, 180, DataType<int>::type);

	for (int i=0; i<180; ++i)
		histogram1.at<int>(0,i) = i,
		histogram2.at<int>(0,i) = i*10;
	
	getDiff();

	// std::cout << res << std::endl;

	return 0;
}

void getDiff(void)
{
	const cv::Mat image1 = cv::imread("pano1.jpg");
	const cv::Mat image2 = cv::imread("pano1_mod.jpg");
	cv::Mat hist;
	cv::Mat difMat = image1.clone();
	// difMat.create(image1.rows, image1.cols, DataType<int>::type);
	
	const int blockSize = 90;
	int difRes =false;

	for (int i=0; i<image1.cols-blockSize; i+=(blockSize/3)) { // traverse columns
		for (int j=0; j<image1.rows-blockSize; j+=(blockSize/3)) { // traverse rows
			
			cv::Rect rectangle1(i, j, blockSize, blockSize);
			cv::Rect rectangle2(i, j, blockSize, blockSize);
			cv::Mat block1 = image1(rectangle1);
			cv::Mat block2 = image2(rectangle2);
			
			cv::Mat histogram1,
					histogram2;

			// cv::imshow("block1", block1);
			// cv::imshow("block2", block2);
			// cv::waitKey(0);

			histogram1.create(1, 180, DataType<int>::type);
			histogram2.create(1, 180, DataType<int>::type);

			getHOGFeatures1(block1, histogram1);
			getHOGFeatures1(block2, histogram2);

			// cv::normalize(histogram1, histogram1);
			// cv::normalize(histogram2, histogram2);

			difRes = findHistogramDif(histogram1, histogram2);
			
			if (difRes == true) {
				// @ref http://stackoverflow.com/questions/10991523/opencv-draw-an-image-over-another-image
				cv::Rect roi(i, j, blockSize, blockSize);
				block2.copyTo(difMat(roi));
			} else {
				for (int ii=i; ii<i+blockSize; ++ii)
					for (int jj=j; jj<j+blockSize; ++jj)
						difMat.at<int>(jj,ii) = 0;
			} // end of if else block

			// cv::imshow("image block", block1), cv::waitKey(0);
		}
	}

	cv::imwrite("dif.jpg", difMat);
	return;
}

bool findHistogramDif(cv::Mat hist1, cv::Mat hist2)
{
	if (hist1.rows != 1 || hist1.cols != 180 || hist2.rows != 1 || hist2.cols != 180) {
		std::cerr << "Histogram matris's size is not proper\n";
		return false;
	}

	int diff =0;
	int diffHistogram =0;

	for (int i=0; i<hist1.cols; ++i) {
		diff = (int) std::abs(hist1.at<int>(0,i) - hist2.at<int>(0,i));

		if (diff>5)
			++diffHistogram;
	}

	if (diffHistogram>50)
		return true;

	return false;
}

void getHOGFeatures1(Mat InputImage, Mat& Histogram) {
	Mat gradH, gradV, imageO, imageM;
	cv::Mat image1= InputImage.clone();

	// cv::Mat kernel;
	// kernel = cv::Mat::ones(5,5,CV_32F)/(float)(5*5);
	// filter2D(InputImage, image1, -1, kernel);

	Sobel(image1, gradH, DataType<float>::type, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT);
	Sobel(image1, gradV, DataType<float>::type, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT);

	imageM.create(InputImage.rows, InputImage.cols, DataType<float>::type);
	imageO.create(InputImage.rows, InputImage.cols, DataType<float>::type);

	// calculate magnitude and orientation images...
	float maxM = 0;
	int r, c;
	for (r=0;r<image1.rows;r++) {
		for (c=0;c<image1.cols;c++) {
			imageO.at<float>(r,c) = (float)(atan2(gradV.at<float>(r,c),gradH.at<float>(r,c)));
			imageM.at<float>(r,c) = gradH.at<float>(r,c)*gradH.at<float>(r,c) + gradV.at<float>(r,c)*gradV.at<float>(r,c);
			if (imageM.at<float>(r,c)>maxM) maxM = imageM.at<float>(r,c);
		}
	}

	// normalize magnitude image to 1...
	for (r=0;r<image1.rows;r++) {
		for (c=0;c<image1.cols;c++) {
			imageM.at<float>(r,c) /= maxM;
		}
	}

	// form the histogram - will get rid of small magnitude orientations
	Histogram.create(1, 180, DataType<int>::type);
	for(c=0; c<Histogram.cols; c++) { 
		Histogram.at<int>(0,c) = 0;
	}

	float stepSize = (float)(2.0*PI/(float)Histogram.cols);
	for (r=3;r<image1.rows-3;r++) {
		for (c=3;c<image1.cols-3;c++) {
			if (imageM.at<float>(r,c)>MINIMUM_GRAD_MAGNITUDE_FOR_ORIENTATION) {
				float theta = imageO.at<float>(r,c); // between -pi and pi...
				theta += (float)PI;
				int count = (int)(theta / stepSize);
				if (count>=0 && count<Histogram.cols) Histogram.at<int>(0,count) += 1;
			}			
		}
	}

	FILE *fileHist = fopen("hist.txt","w");
	
	for(c=0; c<Histogram.cols; c++) {
		fprintf(fileHist, "%d ", Histogram.at<int>(0,c));
	}
	
	fprintf(fileHist, "\n");
	fclose(fileHist);

	// imshow("Orient Image", imageO);
	// imshow("Magnit Image", imageM);
	// cv::waitKey(0);

} // end-getHOGFeatures1