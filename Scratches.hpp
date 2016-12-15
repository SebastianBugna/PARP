// resuelvo con guardas multiples inclusiones
#ifndef HEAP_PUBLICAS_H
#define HEAP_PUBLICAS_H


#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/photo/photo.hpp"
#include <iostream>
#include <algorithm>
#include <vector>

bool sortcol( const std::vector<float>& v1, const std::vector<float>& v2 );
bool sortcol2( const std::vector<float>& v1, const std::vector<float>& v2 ) ;
float NFA(const std::vector<int> perfil, const std::vector<cv::Point> coordenadas, const int c, const int f, const cv::Mat  PM , const long long int  Ntests );

void RestoreScratches(cv::Mat src, cv::Mat &dst, bool detectionMap,  bool original,  bool restored, int thresholdHough, int inclination, int inpaintingRadius, int inpaintingMethod);

void BinaryDetection(cv::Mat src, cv::Mat &dst);
void PixelDensity(cv::Mat bin, cv::Mat &PM);


#endif