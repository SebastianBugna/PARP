


#include <climits>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/photo/photo.hpp"
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

#define kPluginName "Example"
#define kPluginGrouping "ExamplePARP"
#define kPluginDescription "Example plugin, using OpenCV."
#define kPluginIdentifier "net.sf.openfx.Example"
#define kPluginVersionMajor 1 // Incrementing this number means that you have broken backwards compatibility of the plug-in.
#define kPluginVersionMinor 0 // Increment this when you have fixed a bug or made it faster.

#define kSupportsTiles 0
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 1
#define kRenderThreadSafety eRenderFullySafe

#define kParamAdvanced "advanced"
#define kParamAdvancedLabel "Advanced"

#define kParamgenericNumber "Value"
#define kParamgenericNumberLabel "Value"
#define kParamgenericNumberHint "Value for the OpenCV threshold function."

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

class Example : public OpenCVPlugin
{
public:
    /** @brief ctor */
    Example(OfxImageEffectHandle handle) : OpenCVPlugin(handle) , _genericNumber(0)

    {  _genericNumber = fetchDoubleParam(kParamgenericNumber); }

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

    bool computeWindow(const OFX::Image* srcImg, double time, OfxRectI *analysisWindow);

    void weights_function(int p, float* weights);

    OFX::DoubleParam* _genericNumber;

};

template <class PIX>
void Example::readOFX_toOpenCV(const OFX::Image* srcImage, const OfxRectI & renderWindow, cv::Mat &CVSrcMat) {

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


template <class PIX, int maxValue>
void Example::PixelProcessor(   const OFX::RenderArguments &args,
                                OFX::Clip *_srcClip,
                                const OfxPointD& renderScale,
                                const OfxRectI & renderWindow,
                                OFX::Image* dstImage,
                                int srcComponentCount
                              )
{

    CvSize imageSize = cvSize(renderWindow.x2-renderWindow.x1, renderWindow.y2-renderWindow.y1);

    assert(dstImage->getPixelComponents() == OFX::ePixelComponentRGBA);

    Mat CVSrcMat = Mat::zeros(imageSize, CV_32FC3); // mapeo adentro exactamente lo que tiene natron en src;
    std::auto_ptr<const OFX::Image> src((_srcClip && _srcClip->isConnected()) ?  _srcClip->fetchImage(args.time) : 0);
    const OFX::Image* srcImage = src.get();
    readOFX_toOpenCV<float>(srcImage, renderWindow, CVSrcMat);
    
    Mat CVSrcMat_GREY, CVdstMat_GREY, CVdstMat;
    double factor= _genericNumber->getValueAtTime(args.time);
    
    cvtColor( CVSrcMat, CVSrcMat_GREY, CV_BGR2GRAY );
    threshold(CVSrcMat_GREY, CVdstMat_GREY, factor, 255, 0);
    cvtColor( CVdstMat_GREY, CVdstMat, CV_GRAY2BGR );
    
    CVdstMat.convertTo(CVdstMat, CV_32FC3, 1.0/255, 0);

    CVdstMat_GREY.release();

    assert(dstImage->getPixelComponents() == OFX::ePixelComponentRGBA);
    

    switch (   srcImage->getPixelComponents() ) {

        case (OFX::ePixelComponentRGBA) : {

            for(int y = renderWindow.y1; y < renderWindow.y2; y++) {

                PIX *dstPix = (PIX *) dstImage->getPixelAddress(renderWindow.x1, y);
                float *dstPix_Mat = CVdstMat.ptr<float>(y);

                for(int x = renderWindow.x1; x < renderWindow.x2; x++) {
                        
                        dstPix[0] = dstPix_Mat[0];
                        dstPix[1] = dstPix_Mat[1];
                        dstPix[2] = dstPix_Mat[2];
                        dstPix[3] = (float)1;
                        dstPix+=4;
                        dstPix_Mat+=3;

                }

            } 

            CVdstMat.release();
            break;
        } //end RGBA


        case (OFX::ePixelComponentRGB) : {


            for(int y = renderWindow.y1; y < renderWindow.y2; y++) {
                PIX *dstPix = (PIX *) dstImage->getPixelAddress(renderWindow.x1, y);
                float *dstPix_Mat = CVdstMat.ptr<float>(y);

                for(int x = renderWindow.x1; x < renderWindow.x2; x++) {
                        
                        dstPix[0] = dstPix_Mat[0];
                        dstPix[1] = dstPix_Mat[1];
                        dstPix[2] = dstPix_Mat[2];
                        dstPix[3] = (float)1;
                        dstPix+=4;
                        dstPix_Mat+=3;                  
            
                }

            }
            break;
    } // end case RGB

} //end SWITCH

    CVdstMat.release();

} // PixelProcessor


// the overridden render function
void
Example::render(const OFX::RenderArguments &args)
{
    std::auto_ptr<OFX::Image> dst( _dstClip->fetchImage(args.time) );

    if ( !dst.get() ) { OFX::throwSuiteStatusException(kOfxStatFailed);  }

    if ( (dst->getRenderScale().x != args.renderScale.x) ||
         ( dst->getRenderScale().y != args.renderScale.y) ||
         ( dst->getField() != args.fieldToRender) ) {
        setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    int srcComponentCount = _srcClip->getPixelComponentCount();

    PixelProcessor<float, 1>(args, _srcClip, args.renderScale, args.renderWindow, dst.get(), srcComponentCount );
    
} // render



void
Example::getFramesNeeded(const OFX::FramesNeededArguments &args, OFX::FramesNeededSetter &frames) {}

mDeclarePluginFactory(ExampleFactory, {}, {});


using namespace OFX;
void
ExampleFactory::describe(OFX::ImageEffectDescriptor &desc)
{

    
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

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
ExampleFactory::describeInContext(OFX::ImageEffectDescriptor &desc,
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
        DoubleParamDescriptor* param = desc.defineDoubleParam(kParamgenericNumber);
        param->setLabel(kParamgenericNumberLabel);
        param->setHint(kParamgenericNumberHint);
        param->setRange(0, 1);
        param->setDisplayRange(0, 1);
        //param->setIncrement(2.0);
        param->setDefault(0.5);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    
} // describeInContext

OFX::ImageEffect*
ExampleFactory::createInstance(OfxImageEffectHandle handle,
                                             OFX::ContextEnum /*context*/)
{
    return new Example(handle);
}



void
Example::changedParam(const OFX::InstanceChangedArgs &args,
                                    const std::string &paramName)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    bool doUpdate = false;

    OfxRectI analysisWindow;
    const double time = args.time;


} 

static ExampleFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
mRegisterPluginFactoryInstance(p)