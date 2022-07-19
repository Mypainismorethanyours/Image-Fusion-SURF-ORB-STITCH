#include "stdafx.h"
#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>  

using namespace cv;
using namespace std;

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);
void CalcCorners(const Mat& H, const Mat& src);

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

int main(int argc, char *argv[])
{
	Mat image01 = imread("right.jpg", 1);    //right image
	Mat image02 = imread("left.jpg", 1);    //left image

	//Gray image conversion
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);

	//Extracting feature points
	SurfFeatureDetector Detector(2000);
	vector<KeyPoint> keyPoint1, keyPoint2;
	Detector.detect(image1, keyPoint1);
	Detector.detect(image2, keyPoint2);

	//Feature point description to prepare for feature point matching on the lower side
	SurfDescriptorExtractor Descriptor;
	Mat imageDesc1, imageDesc2;
	Descriptor.compute(image1, keyPoint1, imageDesc1);
	Descriptor.compute(image2, keyPoint2, imageDesc2);

	FlannBasedMatcher matcher;
	vector<vector<DMatch> > matchePoints;
	vector<DMatch> GoodMatchePoints;

	vector<Mat> train_desc(1, imageDesc1);
	matcher.add(train_desc);
	matcher.train();

	matcher.knnMatch(imageDesc2, matchePoints, 2);

	// use Lowe's algorithm to get excellent matching points
	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}

	Mat first_match;
	drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
	imwrite("first_match.jpg", first_match);
	cout << "The rist matching result has beed saved."<< endl;

	vector<Point2f> imagePoints1, imagePoints2;

	for (int i = 0; i<GoodMatchePoints.size(); i++)
	{
		imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
		imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
	}

	//get the projection mapping matrix from image 1 to image 2 with size 3¡Á3
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	////can also use the getperspectivetransform method to obtain the perspective transformation matrix, but only four points are required, and the effect is slightly worse
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  

	//Calculate the coordinates of the four vertices of the registration graph
	CalcCorners(homo, image01);

	//image registration  
	Mat imageTransform1, imageTransform2;
	warpPerspective(image01, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
	//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));


	//In order to create a spliced image, the size of the graph needs to be calculated in advance
	int dst_width = imageTransform1.cols;  //Take the length of the rightmost point as the length of the spliced image 
	int dst_height = image02.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	image02.copyTo(dst(Rect(0, 0, image02.cols, image02.rows)));

	imwrite("simple_result.jpg", dst);
	cout << "The simple result has beed saved." << endl;

	OptimizeSeam(image02, imageTransform1, dst);

	imwrite("optimized_result.jpg", dst);
	cout << "The optimized result has beed saved." << endl;

	waitKey();

	return 0;
}

void CalcCorners(const Mat& H, const Mat& src){
	double v2[] = { 0, 0, 1 };//top left corner
	double v1[3];//Coordinate value after transformation
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //Column vector
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //Column vector

	V1 = H * V2;
	//top left corner(0,0,1)
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//lower left corner(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //Column vector
	V1 = Mat(3, 1, CV_64FC1, v1);  //Column vector
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//top right corner(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //Column vector
	V1 = Mat(3, 1, CV_64FC1, v1);  //Column vector
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//Lower right corner(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //Column vector
	V1 = Mat(3, 1, CV_64FC1, v1);  //Column vector
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];
}

//Optimize the connection between the two figures to make the splicing natural
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//The starting position which is the left boundary of the overlapping area

	double processWidth = img1.cols - start;//Width of overlapping area
	int rows = dst.rows;
	int cols = img1.cols; //Note that is the number of columns * the number of channels
	double alpha = 1;//Weight of pixels in img1
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //Get the first address of ith line
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//If a black spot without pixels in the image Trans is encountered, the data in img1 is completely copied
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//The weight of pixels in img1 is directly proportional to the distance between the current processing point and the left boundary of the overlapping area
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}

