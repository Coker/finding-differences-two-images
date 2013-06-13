#define MINIMUM_GRAD_MAGNITUDE_FOR_ORIENTATION 0.05
#define PI 3.14159265
void getHOGFeatures1(Mat InputImage, Mat & Histogram) {
	Mat gradH, gradV, imageO, imageM;

	Sobel(InputImage, gradH, DataType<float>::type, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT);
	Sobel(InputImage, gradV, DataType<float>::type, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT);

	imageM.create(InputImage.rows, InputImage.cols, DataType<float>::type);
	imageO.create(InputImage.rows, InputImage.cols, DataType<float>::type);

	// calculate magnitude and orientation images...
	float maxM = 0;
	int r, c;
	for (r=0;r<InputImage.rows;r++) {
		for (c=0;c<InputImage.cols;c++) {
			imageO.at<float>(r,c) = (float)(atan2(gradV.at<float>(r,c),gradH.at<float>(r,c)));
			imageM.at<float>(r,c) = gradH.at<float>(r,c)*gradH.at<float>(r,c) + gradV.at<float>(r,c)*gradV.at<float>(r,c);
			if (imageM.at<float>(r,c)>maxM) maxM = imageM.at<float>(r,c);
		}
	}
	// normalize magnitude image to 1...
	for (r=0;r<InputImage.rows;r++) {
		for (c=0;c<InputImage.cols;c++) {
			imageM.at<float>(r,c) /= maxM;
		}
	}

	// form the histogram - will get rid of small magnitude orientations
	Histogram.create(1, 180, DataType<int>::type);
	for(c=0; c<Histogram.cols; c++) {
		Histogram.at<int>(0,c) = 0;
	}
	float stepSize = (float)(2.0*PI/(float)Histogram.cols);
	for (r=3;r<InputImage.rows-3;r++) {
		for (c=3;c<InputImage.cols-3;c++) {
			if (imageM.at<float>(r,c)>MINIMUM_GRAD_MAGNITUDE_FOR_ORIENTATION) {
				float theta = imageO.at<float>(r,c); // between -pi and pi...
				theta += (float)PI;
				int count = (int)(theta / stepSize);
				if (count>=0 && count<Histogram.cols) Histogram.at<int>(0,count) += 1;
			}
			else {
			}
		}
	}

	//imshow("Orient Image", imageO); imshow("Magnit Image", imageM); cv::waitKey(0);

	FILE * fileHist = fopen("hist.txt","w");
	for(c=0; c<Histogram.cols; c++) {
		fprintf(fileHist, "%d ", Histogram.at<int>(0,c));
	}
	fprintf(fileHist, "\n");
	fclose(fileHist);
} // end-getHOGFeatures1