#include <iostream>
#include "Scratches.hpp"

using namespace cv;
using namespace std;

void ReadOFXtoOpenCV(cv::Mat OFXSrcImage, cv::Mat &CVSrcMat);

void WriteOpenCVtoOFX(cv::Mat &CVDstMat,cv::Mat &OFXDstImage);

void ReadOFXtoOpenCV(cv::Mat OFXSrcImage, cv::Mat &CVSrcMat)
{
    int nCols = OFXSrcImage.cols;
    int nRows = OFXSrcImage.rows;
    

    CVSrcMat = Mat::zeros(nRows,nCols, CV_32FC3); //no hace nada, es solo para kachegrind
    //cout << "CVRSMat type enseguida" << CVSrcMat.type() << endl;


    for(int y = 0; y < nRows; y++) 
    {
        //PIX *srcPix = (PIX *) srcImage->getPixelAddress(x1, y);
        float *srcPix = OFXSrcImage.ptr<float>(y);

        //uchar *srcPix_openCV = CVSrcMat.ptr<uchar>(y);

        for(int x = 0; x < nCols; x++) 
        {
        	
            CVSrcMat.at<Vec3f>(y,x)[0] = (float)srcPix[0];
            CVSrcMat.at<Vec3f>(y,x)[1] = (float)srcPix[1];
            CVSrcMat.at<Vec3f>(y,x)[2] = (float)srcPix[2] ;
            srcPix=srcPix+3;
            
            /*
            srcPix_openCV[0] = srcPix[0]*255 ;
            srcPix_openCV[1] = srcPix[1]*255 ;
            srcPix_openCV[2] = srcPix[2]*255 ;
            srcPix+=3;
            srcPix_openCV+=3;
            */
            

        }


    }

    //cout << "CVRSMat type adentro" << CVSrcMat.type() << endl;

    CVSrcMat.convertTo(CVSrcMat, CV_8U, 255.0, 0);
}


void WriteOpenCVtoOFX(cv::Mat &CVDstMat,cv::Mat &OFXDstImage)
{
    int nCols = CVDstMat.cols;
    int nRows = CVDstMat.rows;

    Mat CVDstMat_aux = Mat::zeros(nRows,nCols, CV_32FC3); //no hace nada, es solo para kachegrind
    //Mat CVDstMat;

    CVDstMat.convertTo(CVDstMat, CV_32FC3, 1.0/255, 0);
    for(int y = 0; y < nRows; y++) 
    {

        //PIX *dstPix = (PIX *) dstImage->getPixelAddress(renderWindow.x1, y);
        float *dstPix = OFXDstImage.ptr<float>(y);
        for(int x = 0; x < nCols; x++) 
        {
            dstPix[0] = CVDstMat.at<Vec3f>(y,x)[0]; 
            dstPix[1] = CVDstMat.at<Vec3f>(y,x)[1];
            dstPix[2] = CVDstMat.at<Vec3f>(y,x)[2];
            dstPix[3] = (float)1;
            
            dstPix=dstPix+4;
        }
    }


}

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

    cv::Mat frame;

    double rate = capture.get(CV_CAP_PROP_FPS);
    int delay = 1000/rate; //ESTO ERA 1000/rate

    while(true)
    {
        if(!capture.read(frame)){
            break;
        }

        int nCols = frame.cols;
        int nRows = frame.rows;
        int channels = frame.channels();
        int nComponents=4;

        Mat OFXimage;
        frame.convertTo(OFXimage, CV_32FC4, 1.0/255, 0); //Imita Imagen OFX, float, 4 canales, 32 bits
        

        Mat CVSrcMat = Mat::zeros(nRows,nCols,  CV_32FC3); //mapeo adentro lo que tiene natron en src;
        ReadOFXtoOpenCV(OFXimage, CVSrcMat);

        //cvtColor(frame, frame, CV_BGR2GRAY); 
        //cout <<"frame "<< frame.type() << endl;
        //cout <<"CVSrcMat"<< CVSrcMat.type() << endl;

        int original =0; 
        int detectionMap = 0; //Muestra mapa de deteccion
        int restored = 1; //Muestra restauracion inpainting
        int thresholdHough=120; //humbral Hough
        int inclination=1;
        //int scratchWidth=3; //width of the scratch in pixels
        //int inpaintingMapDilate=1;
        int inpaintingRadius=2;
        int inpaintingMethod=1;

        Mat CVDstMat;
        RemoveScratches(CVSrcMat, CVDstMat, 
                        detectionMap,  
                        original,  
                        restored, 
                        thresholdHough, 
                        inclination, 
                        inpaintingRadius, 
                        inpaintingMethod);
                        
        Mat OFXDstImage=Mat::zeros(nRows, nCols, CV_32FC4);;
        WriteOpenCVtoOFX(CVDstMat,OFXDstImage);

     
      //DESPLEIGO RESULTADOS
        namedWindow("ORIGINAL");
        imshow("ORIGINAL",CVSrcMat);
        moveWindow("ORIGINAL", 0, 0);
        namedWindow("SCRATCHES");
        //imshow("SCRATCHES",dst);
        imshow("SCRATCHES",OFXDstImage);
        moveWindow("SCRATCHES", nCols+50, 0);
     

        if(waitKey(delay)>=0)
            break;
    }

    capture.release();

    return 0;

}
