#include <iostream>
#include "Scratches.hpp"

using namespace cv;
using namespace std;

int main(int, char**)
{
    VideoCapture capture("Knight.avi");
    //VideoCapture capture("laurel_hardy.avi");
    //VideoCapture capture("Afgr.avi");
	//VideoCapture capture("JuegosyRondas_HDchico.mov");
    if(!capture.isOpened()){
        std::cout<<"cannot read video!\n";
        return -1;
    }

    Mat frame;
    //namedWindow("frame");

    double rate = capture.get(CV_CAP_PROP_FPS);
    int delay = 1000/rate; //ESTO ERA 1000/rate

    while(true)
    {
        if(!capture.read(frame)){
            break;
        }

        cvtColor(frame, frame, CV_BGR2GRAY);
        Mat src = frame; 

        Mat dst;

        int original =0; 
        int detectionMap = 1; //Muestra mapa de deteccion
        int restored = 0; //Muestra restauracion inpainting
        int thresholdHough=120; //humbral Hough
        int inclination=1;
        //int scratchWidth=3; //width of the scratch in pixels
    	//int inpaintingMapDilate=1;
        int inpaintingRadius=2;
        int inpaintingMethod=1;


        RestoreScratches(src, dst, 
        				detectionMap,  
        				original,  
        				restored, 
        				thresholdHough, 
        				inclination, 
        				inpaintingRadius, 
        				inpaintingMethod);


/*
      //DESPLEIGO RESULTADOS
      namedWindow("ORIGINAL");
      imshow("ORIGINAL",src);
      moveWindow("ORIGINAL", 0, 0);
      namedWindow("SCRATCHES");
      imshow("SCRATCHES",dst);
      int nCols = src.cols;
      moveWindow("SCRATCHES", nCols+50, 0);
*/
       //capture.release(); //SACAR!!!!!!!!!

        if(waitKey(delay)>=0)
            break;
    }

    capture.release();

    return 0;

}