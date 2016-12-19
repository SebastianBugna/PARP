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
float NFA( const std::vector<int> perfil, const std::vector<cv::Point> coordenadas, const int c, const int f, const cv::Mat  PM , const long long int  Ntests );

void RemoveScratches(const cv::Mat src, cv::Mat &dst, bool detectionMap,  bool original,  bool restored, int thresholdHough, int inclination, int inpaintingRadius, int inpaintingMethod);

void BinaryDetection(const cv::Mat src_bw, cv::Mat &dst);

void HoughSpeedUp(const cv::Mat bin, const int thresholdHough, const int inclination, std::vector<std::vector<float> > &lines_Hough);

void MaximalMeaningfulScratchGrouping(std::vector<std::vector<float> > &Detecciones_MAX, const cv::Mat bin, const cv::Mat PM, std::vector<std::vector<float> > lines_Hough,  const long long int  Ntests, const int largo_min);

void Maximality(std::vector<std::vector<float> > &Detecciones, std::vector<std::vector<float> > &Detecciones_MAX);

void PixelDensity(cv::Mat bin, cv::Mat &PM);

void ExclusionPrinciple(const std::vector<std::vector<float> > Detecciones_MAX, std::vector<std::vector<float> > &Detecciones_EXC, const cv::Mat bin, const cv::Mat PM, const long long int  Ntests, const int largo_min );



#endif
