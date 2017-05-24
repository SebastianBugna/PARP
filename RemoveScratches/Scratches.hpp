/*
   OFX RemoveScratches OpenCV plugin.

	PARP: Plataforma Abierta de Restauración de Películas.
	Developed by Sebastián Bugna and Juan Andrés Friss de Kereki.
	Montevideo, Uruguay. Facultad de Ingeniería.

   Copyright (C) 2014 INRIA
   Redistribution and use in source and binary forms, with or without modification,
   are permitted provided that the following conditions are met:
   Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or
   other materials provided with the distribution.
   Neither the name of the {organization} nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
   INRIA
 */

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
    eOverlayDetection,
    eDetectionMask,
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

void ExclusionPrinciple( std::vector<std::vector<float> > &Detecciones_EXC, std::vector<std::vector<float> > &Detecciones_EXCOUT, const cv::Mat bin, const cv::Mat PM, const int nfaThreshold, const long long int  Ntests, const int minLength, const int minDistance );


#endif
