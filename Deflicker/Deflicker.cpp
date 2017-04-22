//#include "GenericOpenCVPlugin.h"

//#include <ofxsLut.h>
#include <climits>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/photo/photo.hpp"
//#include <iostream>
//#include <vector>

#include "ofxsCoords.h"
#include "ofxsProcessing.H"
#include "ofxsMacros.h"
#include "ofxsCopier.h"
#include "ofxsPixelProcessor.h"
#include "ofxsImageEffect.h"



#ifdef OFX_USE_MULTITHREAD_MUTEX
namespace {
typedef OFX::MultiThread::Mutex Mutex;
typedef OFX::MultiThread::AutoMutex AutoMutex;
}
#else
// some OFX hosts do not have mutex handling in the MT-Suite (e.g. Sony Catalyst Edit)
// prefer using the fast mutex by Marcus Geelnard http://tinythreadpp.bitsnbites.eu/
#include "fast_mutex.h"
namespace {
typedef tthread::fast_mutex Mutex;
typedef OFX::MultiThread::AutoMutexT<tthread::fast_mutex> AutoMutex;
}
#endif

#define kPluginName "DeflickerOFX"
#define kPluginGrouping "PARP"
#define kPluginDescription "Remove flicker in old film footage, using OpenCV."
#define kPluginIdentifier "net.sf.openfx.Deflicker"
#define kPluginVersionMajor 1 // Incrementing this number means that you have broken backwards compatibility of the plug-in.
#define kPluginVersionMinor 0 // Increment this when you have fixed a bug or made it faster.

#define kSupportsTiles 0
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 1
#define kRenderThreadSafety eRenderFullySafe

#define kParamAdvanced "advanced"
#define kParamAdvancedLabel "Advanced"

#define kParamgenericNumber "Temporal Window"
#define kParamgenericNumberLabel "Temporal Window"
#define kParamgenericNumberHint "Amount of frames the algorithm takes into account."

using namespace OFX;
using namespace cv;
using namespace std;

class OpenCVPlugin
    : public OFX::ImageEffect
{
public:
    /** @brief ctor */
    OpenCVPlugin(OfxImageEffectHandle handle);

private:
    

protected:

    // do not need to delete these, the ImageEffect is managing them for us
    OFX::Clip *_dstClip;
    OFX::Clip *_srcClip;
};

OpenCVPlugin::OpenCVPlugin(OfxImageEffectHandle handle)
    : ImageEffect(handle)
      , _dstClip(0)
      , _srcClip(0)
{
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
}

class Deflicker : public OpenCVPlugin
{
public:
    /** @brief ctor */
    Deflicker(OfxImageEffectHandle handle) : OpenCVPlugin(handle) , _genericNumber(0)

    {  _genericNumber = fetchIntParam(kParamgenericNumber); }

private:
    /* Override the render */
    virtual void render(const OFX::RenderArguments &args) OVERRIDE FINAL;

    virtual void changedParam(const InstanceChangedArgs &args, const std::string &paramName) OVERRIDE FINAL;

    /** Override the get frames needed action */
    virtual void getFramesNeeded(const OFX::FramesNeededArguments &args, OFX::FramesNeededSetter &frames) OVERRIDE FINAL;

    template <class PIX>
    void readOFX_toOpenCV(const OFX::Image* srcImage, const OfxRectI & renderWindow, cv::Mat &CVSrcMat);

    cv::Mat _histogram_index(const OFX::RenderArguments &args, const OfxRectI & renderWindow, OFX::Clip *_srcClip, int _time, CvSize imageSize, bool histogram);

    template <class PIX, int maxValue>
    void PixelProcessor( const OFX::RenderArguments &args,
                         OFX::Clip *srcClip,
                         const OfxPointD & renderScale,
                         const OfxRectI & renderWindow,
                         OFX::Image* dst,
                         int srcComponentCount
                         );

    //virtual void getRegionsOfInterest(const OFX::RegionsOfInterestArguments &args, OFX::RegionOfInterestSetter &rois) OVERRIDE FINAL;
    //virtual bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments &args, OfxRectD & rod) OVERRIDE FINAL;
    // compute computation window in srcImg
    bool computeWindow(const OFX::Image* srcImg, double time, OfxRectI *analysisWindow);

    void weights_function(int p, float* weights);

    OFX::IntParam* _genericNumber;

};

template <class PIX>
void Deflicker::readOFX_toOpenCV(const OFX::Image* srcImage, const OfxRectI & renderWindow, cv::Mat &CVSrcMat) {

        switch( srcImage->getPixelComponents() ) {

            case OFX::ePixelComponentRGB : { //OFX::ePixelComponentRGB 3 canales
                //cout << "primer case" << endl;
                for(int y = renderWindow.y1; y < renderWindow.y2; y++) {
                    PIX *srcPix       = (PIX *) srcImage->getPixelAddress(renderWindow.x1, y); // puntero de OFX al comienzo de la fila y
                    float *srcPix_Mat = CVSrcMat.ptr<float>(y);                                // punto a la fila y de cvmat
                    for(int x = renderWindow.x1; x < renderWindow.x2*3; x++) {
                        *srcPix_Mat = *srcPix;
                        srcPix++;
                        srcPix_Mat++;
                    }
                } 
                break;
            } 

            case OFX::ePixelComponentRGBA : { //OFX::ePixelComponentRGBA 4 canales
                //cout << "segundo case" << endl;
                for(int y = renderWindow.y1; y < renderWindow.y2; y++) {
                    PIX *srcPix       = (PIX *) srcImage->getPixelAddress(renderWindow.x1, y); // puntero de OFX al comienzo de la fila y
                    float *srcPix_Mat = CVSrcMat.ptr<float>(y);                                // punto a la fila y de cvmat
                    for(int x = renderWindow.x1; x < renderWindow.x2; x++) {
                        srcPix_Mat[0] = srcPix[0];
                        srcPix_Mat[1] = srcPix[1];
                        srcPix_Mat[2] = srcPix[2];
                        srcPix+=4;
                        srcPix_Mat+=3;;
                    }
                } 
                break;
            } 

        } //end switch 
}

void Deflicker::weights_function(int p, float* weights) {

    int var = 2*p; //sigma^2 fijado experimentalmente. paper de referencia : hace mejor el deflicker.
    float sum = 0;
    
    for (int i=0; i<p; i++) {
        weights[i] = exp( - pow ( (i-(p/2)),  2.0)  /  (2*var) );
        sum += weights[i];
    }

    for (int i=0; i<p ; i++){
        weights[i] /= sum; // normalizacion, capaz se puede hacer sin for loop. yo no se como.
        // cout << i << weights[i] << endl;
    }

}

cv::Mat Deflicker::_histogram_index(const OFX::RenderArguments &args, 
                                    const OfxRectI & renderWindow,
                                    OFX::Clip *_srcClip, 
                                    int _time, 
                                    CvSize imageSize, 
                                    bool histogram) { // si histogram, devuelvo histogram, si no index
    
    Mat CVSrcMat = Mat::zeros(imageSize, CV_32FC3); // mapeo adentro exactamente lo que tiene natron en src;
    std::auto_ptr<const OFX::Image> src((_srcClip && _srcClip->isConnected()) ?  _srcClip->fetchImage(args.time + _time) : 0);
    const OFX::Image* srcImage = src.get();
    readOFX_toOpenCV<float>(srcImage, renderWindow, CVSrcMat);

    Mat CVMat, CVMatGray;   
    CVSrcMat.convertTo(CVMat, CV_8U, 255.0, 0);             // CVSRCMAT  - RGB ENTRE 0 Y 1
    cvtColor(CVMat, CVMatGray, CV_RGB2GRAY);                // CVMAT     - RGB ENTRE 0 Y 255
    Mat sameMemoryNewShape = CVMatGray.reshape(1, 1);       // CVMATGRAY - UN CANAL ENTRE 0 Y 255

    //cout <<  "CVMatGray: " << CVMatGray << endl << endl;

    if (histogram) {
        cv::Mat dstSrt = cv::Mat(imageSize,CV_32F);
        cv::sort(sameMemoryNewShape,    dstSrt, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
        //cout <<  "dstSrt: " << dstSrt << endl << endl;
        return dstSrt;
    } else { // index
        cv::Mat dstIdx = cv::Mat(imageSize,CV_32F);
        cv::sortIdx(sameMemoryNewShape, dstIdx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
        //cout <<  "dstIdx: " << dstIdx << endl << endl;
        return dstIdx;
    }
    
} 


template <class PIX, int maxValue>
void Deflicker::PixelProcessor( const OFX::RenderArguments &args,
                                OFX::Clip *_srcClip,
                                const OfxPointD& renderScale,
                                const OfxRectI & renderWindow,
                                OFX::Image* dstImage,
                                int srcComponentCount
                              )
{

    CvSize imageSize = cvSize(renderWindow.x2-renderWindow.x1, renderWindow.y2-renderWindow.y1);
    

    int framesWindow = _genericNumber->getValueAtTime(args.time);
    int radio = (framesWindow-1)/2;
    OfxRangeD frameRange = _srcClip->getFrameRange();
    int minFrame = frameRange.min;
    int maxFrame = frameRange.max;

    float* normal = new float[framesWindow];
    weights_function(framesWindow, normal);
    float* weights = new float[framesWindow];
    float totalWeights = 0;

    for (int i = -radio; i<= radio; i++) {      // es como ir de 0 < framesWindow
        int frameItero = args.time + i;
        if ((frameItero < minFrame) or (frameItero > maxFrame)) {
            weights[i+radio] = 0;               // me paso de los limites del clip
        } else {
            weights[i+radio] = normal[i+radio]; // estoy dentro
        }
        totalWeights += weights[i+radio];
    }
    
    for (int i = 0; i<framesWindow; i++) { // NORMALIZO
        weights[i] != 0 ? weights[i] /= totalWeights : weights[i] = 0;
        //cout << weights[i] << endl;
    }


    int image_size = imageSize.width * imageSize.height;
    double* common_histogram = new double[image_size*100];          // DECLARO EL ARRAY QUE GUARDA HISTOGRAMA COMUN

    for (int i = 0; i < image_size; i++) {
        common_histogram[i] = 0; // limpio
    }    

    for (int i = -radio; i <= radio; i++) { // PARA CADA IMAGEN, i=0 es la actual.. etc
        //cout << "i " << i << " radio " << radio << endl;
        //cout << "peso para la imagen actual> " << weights[i+radio] << endl;
        cv::Mat imgSrt = _histogram_index(args, renderWindow, _srcClip, i, imageSize, true);
        if (weights[i+radio] != 0) {
            for (int j = 0; j < image_size; j++) {
                float product = (float)imgSrt.at<uchar>(0,j)  * (float)weights[i+radio];
                //cout << j << ' ' << (float)imgSrt.at<uchar>(0,j) <<  ' ' << (float)weights[i+radio] << ' ' <<  product << ' ' << common_histogram[j] << ' ' << common_histogram[j] + product << endl;
                common_histogram[j] += product;
            }
            imgSrt.release(); // libero memoria
        }
    }

    cv::Mat dstIdx = _histogram_index(args, renderWindow, _srcClip, 0, imageSize, false); // para la imagen actual, calculo los indices
    Mat CVdstMat = Mat::zeros(imageSize, CV_8U);            // mat de salida
    for (int i = 0; i < image_size; i++) {                  // recorro los vectores
        int index = (int)dstIdx.at<int>(0,i);           // para cada indice
        int _y = index / imageSize.width;
        int _x = index % imageSize.width;                   // calculo x e y correspondientes en la matriz
        CVdstMat.at<uchar>(_y, _x) = common_histogram[i];   // asigno el valor del histograma comun
        //cout << "Index: <x,y> " << index << ":<" << _x << "," << _y << ">" << endl; 
    }

    assert(dstImage->getPixelComponents() == OFX::ePixelComponentRGBA);

    Mat CVSrcMat = Mat::zeros(imageSize, CV_32FC3); // mapeo adentro exactamente lo que tiene natron en src;
    std::auto_ptr<const OFX::Image> src((_srcClip && _srcClip->isConnected()) ?  _srcClip->fetchImage(args.time) : 0);
    const OFX::Image* srcImage = src.get();
    readOFX_toOpenCV<float>(srcImage, renderWindow, CVSrcMat);
    //cout << "cvsrcmat> " << CVSrcMat << endl;

    for(int y = renderWindow.y1; y < renderWindow.y2; y++) {
        PIX *dstPix = (PIX *) dstImage->getPixelAddress(renderWindow.x1, y);
        for(int x = renderWindow.x1; x < renderWindow.x2; x++) {

            dstPix[0] = (float)CVdstMat .at<uchar>(y, x) /255;
            dstPix[1] = (float)CVdstMat .at<uchar>(y, x) /255;
            dstPix[2] = (float)CVdstMat .at<uchar>(y, x) /255;
            dstPix[3] = (float)1;
        
            dstPix=dstPix+4;
        }
    }

    // LIBERO MEMORIA
    dstIdx.release();
    CVdstMat.release();
    delete [] normal; 
    delete [] weights; 
    delete [] common_histogram;
    // LIBERO MEMORIA


} // PixelProcessor


// the overridden render function
void
Deflicker::render(const OFX::RenderArguments &args)
{
    std::auto_ptr<OFX::Image> dst( _dstClip->fetchImage(args.time) );

    if ( !dst.get() ) { OFX::throwSuiteStatusException(kOfxStatFailed);  }

    if ( (dst->getRenderScale().x != args.renderScale.x) ||
         ( dst->getRenderScale().y != args.renderScale.y) ||
         ( dst->getField() != args.fieldToRender) ) {
        setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    //Original source image at current time
    //std::auto_ptr<const OFX::Image> src((_srcClip && _srcClip->isConnected()) ?  _srcClip->fetchImage(args.time) : 0);
    //std::auto_ptr<const OFX::Image> src2((_srcClip && _srcClip->isConnected()) ?  _srcClip->fetchImage(args.time+1) : 0);
    //if ( !src.get() ) { OFX::throwSuiteStatusException(kOfxStatFailed); }

    int srcComponentCount = _srcClip->getPixelComponentCount();


    PixelProcessor<float, 1>(args, _srcClip, args.renderScale, args.renderWindow, dst.get(), srcComponentCount );
    
} // render



void
Deflicker::getFramesNeeded(const OFX::FramesNeededArguments &args, OFX::FramesNeededSetter &frames) {}

mDeclarePluginFactory(DeflickerFactory, {}, {});


using namespace OFX;
void
DeflickerFactory::describe(OFX::ImageEffectDescriptor &desc)
{

    //desc.setOverlayInteractDescriptor(new DeflickerOverlayDescriptor);
    //genericCVDescribe(kPluginName, kPluginGrouping, kPluginDescription, kSupportsTiles, kSupportsMultiResolution, true, kRenderThreadSafety, desc);
	// basic labels
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

    // add the supported contexts
    desc.addSupportedContext(eContextFilter);
    desc.addSupportedContext(eContextGeneral);

    // add supported pixel depths
    desc.addSupportedBitDepth(eBitDepthFloat);

    // set a few flags
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(kSupportsMultiResolution);
    desc.setSupportsTiles(kSupportsTiles);
    desc.setTemporalClipAccess(true);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(kRenderThreadSafety);
}

void
DeflickerFactory::describeInContext(OFX::ImageEffectDescriptor &desc,
                                                OFX::ContextEnum context)
{
    // Source clip only in the filter context
    // create the mandated source clip
    ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);

    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->addSupportedComponent(ePixelComponentRGB);
    srcClip->addSupportedComponent(ePixelComponentAlpha);
    srcClip->setTemporalClipAccess(true);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // create the mandated output clip
    ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);


    // make some pages and to things in
    PageParamDescriptor *page = desc.definePageParam("Controls");

    {
        IntParamDescriptor* param = desc.defineIntParam(kParamgenericNumber);
        param->setLabel(kParamgenericNumberLabel);
        param->setHint(kParamgenericNumberHint);
        param->setRange(1, 99);
        param->setDisplayRange(1, 99);
        //param->setIncrement(2.0);
        param->setDefault(3);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    
} // describeInContext

OFX::ImageEffect*
DeflickerFactory::createInstance(OfxImageEffectHandle handle,
                                             OFX::ContextEnum /*context*/)
{
    return new Deflicker(handle);
}



void
Deflicker::changedParam(const OFX::InstanceChangedArgs &args,
                                    const std::string &paramName)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    bool doUpdate = false;

    OfxRectI analysisWindow;
    const double time = args.time;


} 

static DeflickerFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
mRegisterPluginFactoryInstance(p)