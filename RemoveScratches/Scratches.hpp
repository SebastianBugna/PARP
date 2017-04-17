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

enum OutputEnum // Parametro de interaccion con el usuario. Opcion de video de salida para funcion RemoverScratches. 
{
    eOriginal,
    eDetectionMap,
    eRestoration,
};

enum InpaintingEnum // Parametro de interaccion con el usuario. Opcion de video de salida para funcion RemoverScratches. 
{
    eNavier,
    eFast,
};

bool sortcol( const std::vector<float>& v1, const std::vector<float>& v2 );
bool sortcol2( const std::vector<float>& v1, const std::vector<float>& v2 ) ;
float NFA( const std::vector<int> perfil, const std::vector<cv::Point> coordenadas, const int c, const int f, const cv::Mat  PM , const long long int  Ntests );

void RemoveScratches(const cv::Mat src, cv::Mat &dst,const int nfaThreshold, const int thresholdHough, const int scratchWidth, const int medianDiffThreshold, const int inclination, const int minLength, const int minDistance,const int linesThickness,const int inpaintingRadius, InpaintingEnum inpaintingMethod, OutputEnum output);

void BinaryDetection(const cv::Mat src_bw, cv::Mat &dst, const int scratchWidth, const int medianDiffThreshold);

void HoughSpeedUp(const cv::Mat bin, const int thresholdHough, const int inclination, std::vector<std::vector<float> > &lines_Hough);

void MaximalMeaningfulScratchGrouping(std::vector<std::vector<float> > &Detecciones_MAX, const cv::Mat bin, const cv::Mat PM, const int nfaThreshold, std::vector<std::vector<float> > lines_Hough,  const long long int  Ntests, const int minLength);

void Maximality(std::vector<std::vector<float> > &Detecciones, std::vector<std::vector<float> > &Detecciones_MAX);

void PixelDensity(cv::Mat bin, cv::Mat &PM);

void PixelDensity2(cv::Mat bin, cv::Mat &PM);

void PixelDensity3(cv::Mat bin, cv::Mat &PM);

void ExclusionPrinciple(const std::vector<std::vector<float> > Detecciones_MAX, std::vector<std::vector<float> > &Detecciones_EXC, const cv::Mat bin, const cv::Mat PM, const int nfaThreshold, const long long int  Ntests, const int minLength, const int minDistance );




#endif
