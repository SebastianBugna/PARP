#include "Scratches.hpp"
#include <climits>
#include <algorithm>
#include <cv.h>

#include "ofxsPixelProcessor.h"
#include "ofxsImageEffect.h"
#include "ofxsMacros.h"
#include "ofxsRectangleInteract.h"
#include "ofxsCoords.h"
#include "ofxsProcessing.H"
#include "ofxsMacros.h"
#include "ofxsCopier.h"


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

#define kPluginName "RemoveScratchesOFX"
#define kPluginGrouping "PARP"
#define kPluginDescription "Detect and Restore scratches in degraded films"
#define kPluginIdentifier "net.sf.openfx.RemoveScratches"
#define kPluginVersionMajor 1 // Incrementing this number means that you have broken backwards compatibility of the plug-in.
#define kPluginVersionMinor 0 // Increment this when you have fixed a bug or made it faster.

#define kSupportsTiles 0
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 1
#define kRenderThreadSafety eRenderFullySafe

#define kParamAdvanced "advanced"
#define kParamAdvancedLabel "Advanced"

#define kParamRestrictToRectangle "restrictToRectangle"
#define kParamRestrictToRectangleLabel "Restrict to Rectangle"
#define kParamRestrictToRectangleHint "Restrict scratch detection to a rectangle."

#define kParamAutoUpdate "autoUpdate"
#define kParamAutoUpdateLabel "Auto Update"
#define kParamAutoUpdateHint "Automatically update values when input or rectangle changes if an analysis was performed at current frame. If not checked, values are only updated if the plugin parameters change. "

#define kParamNFAThreshold "nfaThreshold"
#define kParamNFAThresholdLabel "NFA Threshold"
#define kParamNFAThresholdHint "Epsilon-Meaningful Scratches: Values expressed as powers of ten. Thresholds the Number of False Alarms (NFA) over the background model. It is stable around a value of 1. Tune this depending on whether the importance should be put on recall or precision"

#define kParamHough "houghThreshold"
#define kParamHoughLabel "Hough Threshold"
#define kParamHoughHint " Tune slightly to avoid false negative detections."

#define kParamScratchWidth "scratchWidth"
#define kParamScratchWidthLabel "Scratch Width"
#define kParamScratchWidthHint  "Set the width (in pixels) of the scratch. It's automatically set depending on the video resolution. Common values are 3 to 6." 

#define kParamMedianDiffThreshold "medianDiffThreshold"
#define kParamMedianDiffThresholdLabel "Median Diff Threshold"
#define kParamMedianDiffThresholdHint  "s_med: thresholds the difference beetwen the pixel value and it horizontal median. "

#define kParaminclination "inclination"
#define kParaminclinationLabel "Inclination"
#define kParaminclinationHint "Increase to the maximum scratch inclination (in degrees) in your video"

#define kParamMinLength "minLength"
#define kParamMinLengthLabel "Minimum Length"
#define kParamMinLengthHint "Set the minimum scratch length allowed for the detections"

#define kParamMinDistance "minDistance"
#define kParamMinDistanceLabel "Minimum Distance"
#define kParamMinDistanceHint "Exclusion Principle. Set the minimum distance between scratches detected to prune false or redundant detections. "

#define kParamLinesThickness "linesThickness"
#define kParamLinesThicknessLabel "Lines Thickness"
#define kParamLinesThicknessHint "Increase to width of the lines drawed over the detected scratches"

#define kParamOutput "output"
#define kParamOutputLabel "Output Video"
#define kParamOutputHint "First adjust the detection and then the restoration parameters."
#define kParamOutputOptionOriginal "Original"
#define kParamOutputOptionOriginalHint "Use Original to work with the original video"
#define kParamOutputOptionOverlayDetection "Overlay Detections"
#define kParamOutputOptionOverlayDetectionHint "Use to preview the detected scratches on the original image"
#define kParamOutputOptionDetectionMask "Detection Mask"
#define kParamOutputOptionDetectionMaskHint "Use to output the isolated detected scratches as a white mask over black bakground. Useful to use the detections with other inpainting plugins."
#define kParamOutputOptionRestoration "Restoration"
#define kParamOutputOptionRestorationHint "Use to obtain a restored version of the video without the scratches"

#define kParamInpaintingMethod "inpaintingMethod"
#define kParamInpaintingMethodLabel "Inpainting Method"
#define kParamInpaintingMethodHint "Select the desired method from the OpenCV cvInpaint function."
#define kParamInpaintingMethodOptionNavier "Navier-Stokes"
#define kParamInpaintingMethodOptionNavierHint "Image inpainting technique based on fluid dynamic equations Navier-Stokes. (Bertalmio, Bertozzi, Sapiro)"
#define kParamInpaintingMethodOptionFast "Fast Marching"
#define kParamInpaintingMethodOptionFastHint "Image inpainting technique based on the fast marching method. (Alexandru Telea)"

#define kParamInpaintingRadius "inpaintingRadius"
#define kParamInpaintingRadiusLabel "Inpainting Radius"
#define kParamInpaintingRadiusHint "Radius of a circular neighborhood of each point inpainted that is considered by the algorithm"



//#define kParamLeftRightAvgThreshold "leftRightAvgThreshold"
//#define kParamLeftRightAvgThresholdLabel "Left-Right Average Threshold"
//#define kParamLeftRightAvgThresholdHint  "s_avg: thresholds the difference beetwen the averaged pixel values on both sides of the scratch. "

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



class PluginRemoveScratches : public OpenCVPlugin
{
public:
    /** @brief ctor */
    PluginRemoveScratches(OfxImageEffectHandle handle) : OpenCVPlugin(handle) 
    , _btmLeft(0)
    , _size(0)
    , _interactive(0)
    , _restrictToRectangle(0)
    , _nfaThreshold(0)
    , _houghThreshold(0)
    , _scratchWidth(0)
    , _medianDiffThreshold(0)
    , _inclination(0)
    , _minLength(0)
    , _minDistance(0)
    , _linesThickness(0)
    , _output(int(eOriginal))
    , _inpaintingMethod(int(eNavier))
    , _inpaintingRadius(0)
    

    {
        _btmLeft = fetchDouble2DParam(kParamRectangleInteractBtmLeft);
        _size = fetchDouble2DParam(kParamRectangleInteractSize);
        _interactive = fetchBooleanParam(kParamRectangleInteractInteractive);
        _restrictToRectangle = fetchBooleanParam(kParamRestrictToRectangle);
        _autoUpdate = fetchBooleanParam(kParamAutoUpdate);
        _nfaThreshold = fetchIntParam(kParamNFAThreshold);    
        _houghThreshold = fetchIntParam(kParamHough);
        _scratchWidth= fetchIntParam(kParamScratchWidth);
        _medianDiffThreshold= fetchIntParam(kParamMedianDiffThreshold);
        _inclination = fetchIntParam(kParaminclination);      
        _minLength= fetchIntParam(kParamMinLength);
        _minDistance= fetchIntParam(kParamMinDistance);
        _linesThickness = fetchIntParam(kParamLinesThickness);
        _output = fetchChoiceParam(kParamOutput);
        _inpaintingMethod = fetchChoiceParam(kParamInpaintingMethod);
        _inpaintingRadius = fetchIntParam(kParamInpaintingRadius);

        assert(_output);
        assert(_inpaintingMethod);

        assert(_btmLeft && _size && _interactive && _restrictToRectangle && _autoUpdate);

        bool restrictToRectangle = _restrictToRectangle->getValue();
        _btmLeft->setEnabled(restrictToRectangle);
        _btmLeft->setIsSecret(!restrictToRectangle);
        _size->setEnabled(restrictToRectangle);
        _size->setIsSecret(!restrictToRectangle);

        bool doUpdate = _autoUpdate->getValue();
        //bool doDetectionMap =  _detectionMap->getValue()

        _interactive->setEnabled(restrictToRectangle && doUpdate);
        _interactive->setIsSecret(!restrictToRectangle || !doUpdate);

    }

private:


    /* Override the render */
    virtual void render(const OFX::RenderArguments &args) OVERRIDE FINAL;

    virtual void changedParam(const InstanceChangedArgs &args, const std::string &paramName) OVERRIDE FINAL;

    /** Override the get frames needed action */
    virtual void getFramesNeeded(const OFX::FramesNeededArguments &args, OFX::FramesNeededSetter &frames) OVERRIDE FINAL;


    template <class PIX, int maxValue>
    void PixelProcessor(const OFX::Image* ref,
                        const OfxPointD & renderScale,
                        const OfxRectI & renderWindow,
                        OFX::Image* dst,
                        int srcComponentCount,
                        int nfaThreshold,
                        int houghThreshold,
                        int scratchWidth,
                        int medianDiffThreshold,
                        int inclination,
                        int minLength,
                        int minDistance,
                        int linesThickness,
                        OutputEnum output, 
                        InpaintingEnum inpaintingMethod,
                        int inpaintingRadius,
                        OfxRectD regionOfInterest);

    // compute computation window in srcImg
    bool computeWindow(const OFX::Image* srcImg, double time, OfxRectI *analysisWindow);

    Double2DParam* _btmLeft;
    Double2DParam* _size;
    BooleanParam* _interactive;  
    BooleanParam* _restrictToRectangle;
    BooleanParam* _autoUpdate;  
    OFX::IntParam* _nfaThreshold;
    OFX::IntParam* _houghThreshold;
    OFX::IntParam* _scratchWidth;
    OFX::IntParam* _medianDiffThreshold;
    OFX::IntParam* _inclination;
    OFX::IntParam* _minLength;
    OFX::IntParam* _minDistance;
    OFX::IntParam* _linesThickness;
    OFX::ChoiceParam* _output;
    OFX::ChoiceParam* _inpaintingMethod;
    OFX::IntParam* _inpaintingRadius;
};





template <class PIX, int maxValue>
void PluginRemoveScratches::PixelProcessor(   const OFX::Image* srcImage,
                                       const OfxPointD& renderScale,
                                       const OfxRectI & renderWindow,
                                       OFX::Image* dstImage,
                                       int srcComponentCount,
                                       int nfaThreshold,
                                       int houghThreshold,
                                       int scratchWidth,
                                       int medianDiffThreshold,
                                       int inclination,
                                       int minLength,
                                       int minDistance,
                                       int linesThickness,
                                       OutputEnum output,
                                       InpaintingEnum inpaintingMethod,
                                       int inpaintingRadius,
                                       OfxRectD regionOfInterest)
{

// calculamos los limites por si el rectangulo se escapa de la render window
    int lowerBound_y = max( (int)renderWindow.y1, (int)regionOfInterest.y1);
    int lowerBound_x = max( (int)renderWindow.x1, (int)regionOfInterest.x1);
    int maxBound_y   = min( (int)renderWindow.y2, (int)regionOfInterest.y2);
    int maxBound_x   = min( (int)renderWindow.x2, (int)regionOfInterest.x2);

    //CvSize imageSize = cvSize(maxBound_x-lowerBound_x, maxBound_y-lowerBound_y);
    
    CvSize imageSize = cvSize(renderWindow.x2-renderWindow.x1, renderWindow.y2-renderWindow.y1);
    Mat CVSrcMat = Mat::zeros(imageSize, CV_32FC3); //mapeo adentro exactamente lo que tiene natron en src;
    //cout << imageSize << endl;
    switch( srcImage->getPixelComponents() ) {

			case OFX::ePixelComponentRGB : { //OFX::ePixelComponentRGB 3 canales
				//cout << "primer case " << endl;
				for(int y = renderWindow.y1; y < renderWindow.y2; y++) {
			        PIX *srcPix       = (PIX *) srcImage->getPixelAddress(renderWindow.x1, y); // puntero de OFX al comienzo de la fila y
                    srcPix += lowerBound_x*3;
			        float *srcPix_Mat = CVSrcMat.ptr<float>(y);                                // punto a la fila y de cvmat
                    
			        for(int x = renderWindow.x1; x < renderWindow.x2; x++) {

                        srcPix_Mat[0] = srcPix[0] ;
                        srcPix_Mat[1] = srcPix[1] ;
                        srcPix_Mat[2] = srcPix[2] ;
                        srcPix+=3;
                        srcPix_Mat+=3;
			        }
	    		} 
	    		break;
			} 

			case OFX::ePixelComponentRGBA : { //OFX::ePixelComponentRGBA 4 canales
								//cout << "Segundo case " << endl;

				for(int y = renderWindow.y1; y < renderWindow.y2; y++) {
			        PIX *srcPix       = (PIX *) srcImage->getPixelAddress(renderWindow.x1, y); // puntero de OFX al comienzo de la fila y
			        srcPix += lowerBound_x*4;
                    float *srcPix_Mat = CVSrcMat.ptr<float>(y);                                // punto a la fila y de cvmat
                    
			        for(int x = renderWindow.x1; x < renderWindow.x2; x++) {
			            srcPix_Mat[0] = srcPix[0];
			            srcPix_Mat[1] = srcPix[1];
			            srcPix_Mat[2] = srcPix[2];
			            srcPix+=4;
			            srcPix_Mat+=3;
			        }
		    	} 
		    	break;
			} 


		} //end switch

////////////////////////////////////////////////////////////////////////////////////////////
/* FUNCION GENERICA opencv*/////////////////////////////////////////////////////////////////
    
    Mat CVMat, CVMat2;
    CVSrcMat.convertTo(CVMat, CV_8U, 255.0, 0);

    CVSrcMat.release();

    if( output == eOriginal)
    {
        
    CVMat2=CVMat;
    } else {
        RemoveScratches(CVMat,CVMat2, nfaThreshold, houghThreshold, scratchWidth, medianDiffThreshold, inclination, minLength, minDistance, linesThickness, inpaintingRadius, inpaintingMethod, output);
    }  
    

    CVMat.release();
    
    // AHORA CONVIERTO A Floar de 0 a 1 con 4 Bytes por Pixel, con funcion de OpenCV. Para mapar a OFXdstImage
    Mat CVdstMat;

    CVMat2.convertTo(CVdstMat, CV_32FC3, 1.0/255, 0);

    CVMat2.release();

    assert(dstImage->getPixelComponents() == OFX::ePixelComponentRGBA);
    

    switch (   srcImage->getPixelComponents() ) {

    	case (OFX::ePixelComponentRGBA) : {

    		for(int y = renderWindow.y1; y < renderWindow.y2; y++) {

		        PIX *_srcPix = (PIX *) srcImage->getPixelAddress(renderWindow.x1, y);
		        PIX *dstPix = (PIX *) dstImage->getPixelAddress(renderWindow.x1, y);
		        //float *dstPix_Mat = CVdstMat.ptr<float>(y);

		        for(int x = renderWindow.x1; x < renderWindow.x2; x++) {
		            //std::cout << "x  " << x << "  y  " << y << std::endl;
		            if ( (x<regionOfInterest.x2*renderScale.x)&&
		                (x>regionOfInterest.x1*renderScale.x)&&
		                (y<regionOfInterest.y2*renderScale.y)&&
		                (y>regionOfInterest.y1*renderScale.y) ) { // estoy adentro del rectangulo

                        //float *dstPix_Mat = CVdstMat.ptr<float>(y-lowerBound_y);
                        float *dstPix_Mat = CVdstMat.ptr<float>(y);
                        dstPix_Mat += (x-lowerBound_x)*3; 
                        
		                
                        dstPix[0] = dstPix_Mat[0];
                        dstPix[1] = dstPix_Mat[1];
                        dstPix[2] = dstPix_Mat[2];
                        dstPix[3] = (float)1;
                        dstPix+=4;
                        dstPix_Mat+=3;
                        _srcPix+=4;
            			
		            } else {

		                dstPix[0] = (float)_srcPix[0];
		                dstPix[1] = (float)_srcPix[1];
		                dstPix[2] = (float)_srcPix[2];
		                dstPix[3] = (float)_srcPix[3];
		            	_srcPix+=4;
                        dstPix+=4;
                        //dstPix_Mat+=3;
		            
		        	}

    			}

    		} 

            CVdstMat.release();
            break;
    	} //end RGBA


    	case (OFX::ePixelComponentRGB) : {


    		for(int y = renderWindow.y1; y < renderWindow.y2; y++) {
		        PIX *_srcPix = (PIX *) srcImage->getPixelAddress(renderWindow.x1, y);
		        PIX *dstPix = (PIX *) dstImage->getPixelAddress(renderWindow.x1, y);
		        //float *dstPix_Mat = CVdstMat.ptr<float>(y);
		        for(int x = renderWindow.x1; x < renderWindow.x2; x++) {
		            //std::cout << "x  " << x << "  y  " << y << std::endl;
		            if ( (x<regionOfInterest.x2*renderScale.x)&&
		                (x>regionOfInterest.x1*renderScale.x)&&
		                (y<regionOfInterest.y2*renderScale.y)&&
		                (y>regionOfInterest.y1*renderScale.y) ) { // estoy adentro del rectangulo

                        //float *dstPix_Mat = CVdstMat.ptr<float>(y-lowerBound_y);
                        float *dstPix_Mat = CVdstMat.ptr<float>(y);
                        dstPix_Mat += (x-lowerBound_x)*3; 

                        
                        dstPix[0] = dstPix_Mat[0];
                        dstPix[1] = dstPix_Mat[1];
                        dstPix[2] = dstPix_Mat[2];
                        dstPix[3] = (float)1;
                        dstPix+=4;
                        dstPix_Mat+=3;
                        _srcPix+=3;		                
		                
		            } else {

		                dstPix[0] = (float)_srcPix[0];
		                dstPix[1] = (float)_srcPix[1];
		                dstPix[2] = (float)_srcPix[2];
                        dstPix[3] = (float)1;
                        dstPix+=4;
		            	_srcPix+=3;
                        //dstPix_Mat+=3;
		            
		        	}
		        	
    			}

    		}
            break;
    } // end case RGB

} //end SWITCH

    
} // PixelProcessor


// the overridden render function
void
PluginRemoveScratches::render(const OFX::RenderArguments &args)
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
    std::auto_ptr<const OFX::Image> src((_srcClip && _srcClip->isConnected()) ?  _srcClip->fetchImage(args.time) : 0);

    if ( !src.get() ) { OFX::throwSuiteStatusException(kOfxStatFailed); }

    int srcComponentCount = _srcClip->getPixelComponentCount();

    int nfaThreshold = _nfaThreshold->getValueAtTime(args.time);

    int houghThreshold = _houghThreshold->getValueAtTime(args.time);

    int scratchWidth = _scratchWidth->getValueAtTime(args.time);
    
    int medianDiffThreshold = _medianDiffThreshold->getValueAtTime(args.time);

    int inclination = _inclination->getValueAtTime(args.time);

    int minLength = _minLength->getValueAtTime(args.time);

    int minDistance = _minDistance->getValueAtTime(args.time);

    int linesThickness = _linesThickness->getValueAtTime(args.time);

    OutputEnum output = (OutputEnum)_output->getValueAtTime(args.time);

    InpaintingEnum inpaintingMethod = (InpaintingEnum)_inpaintingMethod->getValueAtTime(args.time);

    int inpaintingRadius = _inpaintingRadius->getValueAtTime(args.time);

     

    OfxRectD regionOfInterest;
    _btmLeft->getValueAtTime(args.time, regionOfInterest.x1, regionOfInterest.y1);
    _size->getValueAtTime(args.time, regionOfInterest.x2, regionOfInterest.y2);
    regionOfInterest.x2 += regionOfInterest.x1;
    regionOfInterest.y2 += regionOfInterest.y1;

    //int sizeY=(int)regionOfInterest.y2;
    //_minLength->getValueAtTime(args.time, sizeY);
    //int minLength=30;
    //_minLength->getValueAtTime(args.time, minLength);

        

    PixelProcessor<float, 1>(src.get(), args.renderScale, args.renderWindow, dst.get(), srcComponentCount, nfaThreshold, houghThreshold, scratchWidth, medianDiffThreshold, inclination, minLength, minDistance, linesThickness, output, inpaintingMethod, inpaintingRadius, regionOfInterest );

} // render




void
PluginRemoveScratches::getFramesNeeded(const OFX::FramesNeededArguments &args, OFX::FramesNeededSetter &frames) {}

mDeclarePluginFactory(PluginRemoveScratchesFactory, {}, {});

class PluginRemoveScratchesInteract
    : public RectangleInteract
{
public:

    PluginRemoveScratchesInteract(OfxInteractHandle handle,
                            OFX::ImageEffect* effect)
        : RectangleInteract(handle, effect)
        , _restrictToRectangle(0)
    {
        _restrictToRectangle = effect->fetchBooleanParam(kParamRestrictToRectangle);
        addParamToSlaveTo(_restrictToRectangle);
    }

private:

    // overridden functions from OFX::Interact to do things
    virtual bool draw(const OFX::DrawArgs &args) OVERRIDE FINAL
    {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

        if (restrictToRectangle) {
            return RectangleInteract::draw(args);
        }

        return false;
    }

    virtual bool penMotion(const OFX::PenArgs &args) OVERRIDE FINAL
    {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

        if (restrictToRectangle) {
            return RectangleInteract::penMotion(args);
        }

        return false;
    }

    virtual bool penDown(const OFX::PenArgs &args) OVERRIDE FINAL
    {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

        if (restrictToRectangle) {
            return RectangleInteract::penDown(args);
        }

        return false;
    }

    virtual bool penUp(const OFX::PenArgs &args) OVERRIDE FINAL
    {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

        if (restrictToRectangle) {
            return RectangleInteract::penUp(args);
        }

        return false;
    }

    //virtual bool keyDown(const OFX::KeyArgs &args) OVERRIDE;
    //virtual bool keyUp(const OFX::KeyArgs & args) OVERRIDE;
    //virtual void loseFocus(const FocusArgs &args) OVERRIDE FINAL;


    OFX::BooleanParam* _restrictToRectangle;
};


class PluginRemoveScratchesOverlayDescriptor
    : public DefaultEffectOverlayDescriptor<PluginRemoveScratchesOverlayDescriptor, PluginRemoveScratchesInteract>
{
};

using namespace OFX;


void
PluginRemoveScratchesFactory::describe(OFX::ImageEffectDescriptor &desc)
{

    desc.setOverlayInteractDescriptor(new PluginRemoveScratchesOverlayDescriptor);
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
PluginRemoveScratchesFactory::describeInContext(OFX::ImageEffectDescriptor &desc,
                                                OFX::ContextEnum context)
{
    // Source clip only in the filter context
    // create the mandated source clip
    ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    //OfxPropertySetHandle desc->getPropertySet();
    //PropertySet desc->getPropertySet();
    

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

    
    // restrictToRectangle
    {
        BooleanParamDescriptor *param = desc.defineBooleanParam(kParamRestrictToRectangle);
        param->setLabel(kParamRestrictToRectangleLabel);
        param->setHint(kParamRestrictToRectangleHint);
        param->setDefault(false);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }

    // btmLeft
    {
        Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamRectangleInteractBtmLeft);
        param->setLabel(kParamRectangleInteractBtmLeftLabel);
        param->setDoubleType(eDoubleTypeXYAbsolute);
        param->setDefaultCoordinateSystem(eCoordinatesNormalised);
        param->setDefault(0., 0.);
        param->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
        param->setDisplayRange(-10000, -10000, 10000, 10000); // Resolve requires display range or values are clamped to (-1,1)
        param->setIncrement(1.);
        param->setHint(kParamRectangleInteractBtmLeftHint);
        param->setDigits(0);
        param->setEvaluateOnChange(false);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // size
    {
        Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamRectangleInteractSize);
        param->setLabel(kParamRectangleInteractSizeLabel);
        param->setDoubleType(eDoubleTypeXY);
        param->setDefaultCoordinateSystem(eCoordinatesNormalised);
        param->setDefault(1., 1.);
        param->setRange(0., 0., DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
        param->setDisplayRange(0, 0, 10000, 10000); // Resolve requires display range or values are clamped to (-1,1)
        param->setIncrement(1.);
        param->setDimensionLabels(kParamRectangleInteractSizeDim1, kParamRectangleInteractSizeDim2);
        param->setHint(kParamRectangleInteractSizeHint);
        param->setDigits(0);
        param->setEvaluateOnChange(false);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // autoUpdate
    {
        BooleanParamDescriptor *param = desc.defineBooleanParam(kParamAutoUpdate);
        param->setLabel(kParamAutoUpdateLabel);
        param->setHint(kParamAutoUpdateHint);
        param->setDefault(true);
        param->setAnimates(false);
        if (page) {
            page->addChild(*param);
        }
    }



    // interactive
    {
        BooleanParamDescriptor* param = desc.defineBooleanParam(kParamRectangleInteractInteractive);
        param->setLabel(kParamRectangleInteractInteractiveLabel);
        param->setHint(kParamRectangleInteractInteractiveHint);
        param->setEvaluateOnChange(false);
        if (page) {
            page->addChild(*param);
        }
    }


    // Output Choice Param
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamOutput);
        param->setLabel(kParamOutputLabel);
        param->setHint(kParamOutputHint);
        assert(param->getNOptions() == eOriginal);
        param->appendOption(kParamOutputOptionOriginal, kParamOutputOptionOriginalHint);
        assert(param->getNOptions() == eOverlayDetection);
        param->appendOption(kParamOutputOptionOverlayDetection, kParamOutputOptionOverlayDetectionHint);   
        assert(param->getNOptions() == eDetectionMask);
        param->appendOption(kParamOutputOptionDetectionMask, kParamOutputOptionDetectionMaskHint); 
        assert(param->getNOptions() == eRestoration);
        param->appendOption(kParamOutputOptionRestoration, kParamOutputOptionRestorationHint);
        param->setDefault( int(eOverlayDetection) );
        param->setAnimates(false); // cant animate
        
        if (page) {
            page->addChild(*param);
        }
    }

    // nfaThreshold
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamNFAThreshold);
        param->setLabel(kParamNFAThresholdLabel);
        param->setHint(kParamNFAThresholdHint);
        param->setRange(-50, 50);
        param->setDisplayRange(-20, 20);
        param->setDefault(0);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // houghThreshold
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamHough);
        param->setLabel(kParamHoughLabel);
        param->setHint(kParamHoughHint);
        param->setRange(1, 500);
        param->setDisplayRange(1, 500);
        param->setDefault(120);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // ScratchWidth
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamScratchWidth);
        param->setLabel(kParamScratchWidthLabel);
        param->setHint(kParamScratchWidthHint);
        param->setRange(1, 20);
        param->setDisplayRange(1, 20);
        param->setDefault(5);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // MedianDiffThreshold
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamMedianDiffThreshold);
        param->setLabel(kParamMedianDiffThresholdLabel);
        param->setHint(kParamMedianDiffThresholdHint);
        param->setRange(1, 6);
        param->setDisplayRange(1, 6);
        param->setDefault(3);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // inclination
    {
        IntParamDescriptor* param = desc.defineIntParam(kParaminclination);
        param->setLabel(kParaminclinationLabel);
        param->setHint(kParaminclinationHint);
        param->setRange(1, 20);
        param->setDisplayRange(1, 20);
        param->setDefault(5);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // minLength
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamMinLength);
        param->setLabel(kParamMinLengthLabel);
        param->setHint(kParamMinLengthHint);
        param->setRange(1, 100);
        param->setDisplayRange(1, 100);
        param->setDefault(10);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // minDistance
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamMinDistance);
        param->setLabel(kParamMinDistanceLabel);
        param->setHint(kParamMinDistanceHint);
        param->setRange(0, 20);
        param->setDisplayRange(0, 20);
        param->setDefault(3);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // linesThickness
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamLinesThickness);
        param->setLabel(kParamLinesThicknessLabel);
        param->setHint(kParamLinesThicknessHint);
        param->setRange(1, 10);
        param->setDisplayRange(1, 10);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    // Inpainting Method
    {
        ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamInpaintingMethod);
        param->setLabel(kParamInpaintingMethodLabel);
        param->setHint(kParamInpaintingMethodHint);
        assert(param->getNOptions() == eNavier);
        param->appendOption(kParamInpaintingMethodOptionNavier, kParamInpaintingMethodOptionNavierHint);   
        assert(param->getNOptions() == eFast);
        param->appendOption(kParamInpaintingMethodOptionFast, kParamInpaintingMethodOptionFastHint); 
        param->setDefault( int(eNavier) );
        param->setAnimates(false); // cant animate
        
        if (page) {
            page->addChild(*param);
        }
    }

    // inpaintingRadius
    {
        IntParamDescriptor* param = desc.defineIntParam(kParamInpaintingRadius);
        param->setLabel(kParamInpaintingRadiusLabel);
        param->setHint(kParamInpaintingRadiusHint);
        param->setRange(1, 10);
        param->setDisplayRange(1, 10);
        param->setDefault(4);
        param->setAnimates(true);
        if (page) {
            page->addChild(*param);
        }
    }

    
    
} // describeInContext

OFX::ImageEffect*
PluginRemoveScratchesFactory::createInstance(OfxImageEffectHandle handle,
                                             OFX::ContextEnum /*context*/)
{
    return new PluginRemoveScratches(handle);
}

void
PluginRemoveScratches::changedParam(const OFX::InstanceChangedArgs &args,
                                    const std::string &paramName)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    bool doUpdate = false;
    //bool doDetectionMap = false;

    OfxRectI analysisWindow;
    const double time = args.time;

    if (paramName == kParamRestrictToRectangle) {
        // update visibility
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(time);
        _btmLeft->setEnabled(restrictToRectangle);
        _btmLeft->setIsSecret(!restrictToRectangle);
        _size->setEnabled(restrictToRectangle);
        _size->setIsSecret(!restrictToRectangle);
        _interactive->setEnabled(restrictToRectangle);
        _interactive->setIsSecret(!restrictToRectangle);
        //doUpdate = true;
        //doDetectionMap = false;
    }
    if (paramName == kParamAutoUpdate) {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(time);
        doUpdate = _autoUpdate->getValueAtTime(time);
        _interactive->setEnabled(restrictToRectangle && doUpdate);
        _interactive->setIsSecret(!restrictToRectangle || !doUpdate);
    }
    if (//paramName == kParamRectangleInteractBtmLeft ||
        // only trigger on kParamRectangleInteractSize (the last one changed)
        paramName == kParamRectangleInteractSize) {
        doUpdate = _autoUpdate->getValueAtTime(time);
    }

} // changedParam

static PluginRemoveScratchesFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
mRegisterPluginFactoryInstance(p)