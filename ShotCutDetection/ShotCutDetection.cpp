#include <cmath>
#include <cfloat> // DBL_MAX
#include <climits>
#include <algorithm>
#include <limits>

#include "ofxsProcessing.H"
#include "ofxsRectangleInteract.h"
#include "ofxsMacros.h"
#include "ofxsCopier.h"
#include "ofxsCoords.h"
#include "ofxsLut.h"
#include "ofxsThreadSuite.h"
#include "ofxsMultiThread.h"
#ifdef OFX_USE_MULTITHREAD_MUTEX
namespace {
typedef MultiThread::Mutex Mutex;
typedef MultiThread::AutoMutex AutoMutex;
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

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#include <windows.h>
#define isnan _isnan
#else
using std::isnan;
#endif

using namespace OFX;

OFXS_NAMESPACE_ANONYMOUS_ENTER


#define kPluginName "ShotCutDetectionOFX"
#define kPluginGrouping "PARP"
#define kPluginDescription \
"Detect shotcuts " \
" "
#define kPluginIdentifier "net.sf.openfx.ShotCutDetection"
#define kPluginVersionMajor 1 // Incrementing this number means that you have broken backwards compatibility of the plug-in.
#define kPluginVersionMinor 0 // Increment this when you have fixed a bug or made it faster.

#define kSupportsTiles 1
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 0 // no renderscale support: statistics are computed at full resolution
#define kSupportsMultipleClipPARs false
#define kSupportsMultipleClipDepths false
#define kRenderThreadSafety eRenderFullySafe


#define kParamAnalyzeSequence "analyzeSequence"
#define kParamAnalyzeSequenceLabel "Analyze Sequence"
#define kParamAnalyzeSequenceHint "Analyze all frames from the sequence and set values."

#define kParamClearSequence "clearSequence"
#define kParamClearSequenceLabel "Clear Sequence"
#define kParamClearSequenceHint "Clear analysis for all frames from the sequence."

#define kParamSDA "SDA"
#define kParamSDALabel "SDA"
#define kParamSDAHint "SDA, or Sum of Absolute Differences, is the value of the difference between this frame and the next."

#define kParamThreshold "Threshold"
#define kParamThresholdLabel "Threshold"
#define kParamThresholdHint "Threshold value to create shot cuts."

static bool gHostSupportsDefaultCoordinateSystem = true; // for kParamDefaultsNormalised

#define POINT_TOLERANCE 6
#define POINT_SIZE 5


struct RGBAValues
{
    double r, g, b, a;
    RGBAValues(double v) : r(v), g(v), b(v), a(v) {}

    RGBAValues() : r(0), g(0), b(0), a(0) {}
};

struct Results
{
    RGBAValues mean;
};

class ImageStatisticsProcessorBase
    : public ImageProcessor
{
protected:
    Mutex _mutex; //< this is used so we can multi-thread the analysis and protect the shared results
    unsigned long _count;
    const OFX::Image *_srcImg2;

public:
    ImageStatisticsProcessorBase(ImageEffect &instance)
        : ImageProcessor(instance)
        , _mutex()
        , _count(0)
    {
    }

    virtual ~ImageStatisticsProcessorBase()
    {
    }

    virtual void setPrevResults(const Results &results) = 0;
    virtual void getResults(Results *results) = 0;

    void setSrcImg2(const OFX::Image *v) { _srcImg2 = v; }

protected:

    template<class PIX, int nComponents, int maxValue>
    void toRGBA(const PIX *p,
                RGBAValues* rgba)
    {
        if (nComponents == 4) {
            rgba->r = p[0] / (double)maxValue;
            rgba->g = p[1] / (double)maxValue;
            rgba->b = p[2] / (double)maxValue;
            rgba->a = p[3] / (double)maxValue;
        } else if (nComponents == 3) {
            rgba->r = p[0] / (double)maxValue;
            rgba->g = p[1] / (double)maxValue;
            rgba->b = p[2] / (double)maxValue;
            rgba->a = 0;
        } else if (nComponents == 2) {
            rgba->r = p[0] / (double)maxValue;
            rgba->g = p[1] / (double)maxValue;
            rgba->b = 0;
            rgba->a = 0;
        } else if (nComponents == 1) {
            rgba->r = 0;
            rgba->g = 0;
            rgba->b = 0;
            rgba->a = p[0] / (double)maxValue;
        } else {
            rgba->r = 0;
            rgba->g = 0;
            rgba->b = 0;
            rgba->a = 0;
        }
    }

    template<class PIX, int nComponents, int maxValue>
    void toComponents(const RGBAValues& rgba,
                      PIX *p)
    {
        if (nComponents == 4) {
            p[0] = rgba.r * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
            p[1] = rgba.g * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
            p[2] = rgba.b * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
            p[3] = rgba.a * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
        } else if (nComponents == 3) {
            p[0] = rgba.r * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
            p[1] = rgba.g * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
            p[2] = rgba.b * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
        } else if (nComponents == 2) {
            p[0] = rgba.r * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
            p[1] = rgba.g * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
        } else if (nComponents == 1) {
            p[0] = rgba.a * maxValue + ( (maxValue != 1) ? 0.5 : 0 );
        }
    }
};


template <class PIX, int nComponents, int maxValue>
class ImageMeanProcessor
    : public ImageStatisticsProcessorBase
{
private:
    double _sum[nComponents];

public:
    ImageMeanProcessor(ImageEffect &instance)
        : ImageStatisticsProcessorBase(instance)
    {
        std::fill(_sum, _sum + nComponents, 0.);
    }

    ~ImageMeanProcessor()
    {
    }

    void setPrevResults(const Results &/*results*/) OVERRIDE FINAL {}

    void getResults(Results *results) OVERRIDE FINAL
    {
        if (_count > 0) {
            double mean[nComponents];
            mean[0] = 0;
            for (int c = 0; c < nComponents; ++c) {
                mean[0] += _sum[c];//_count;
            }
            mean[0] = mean[0] / _count;
            //std::cout << mean[0] << std::endl;
            toRGBA<double, nComponents, 1>(mean, &results->mean);
        }
    }

private:

    void addResults(double sum[nComponents],
                    unsigned long count)
    {
        _mutex.lock();
        for (int c = 0; c < nComponents; ++c) {
            _sum[c] += sum[c];
        }
        _count += count;
        _mutex.unlock();
    }

    void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
    {
        double sum[nComponents];

        std::fill(sum, sum + nComponents, 0.);
        unsigned long count = 0;

        assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
               _dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
        for (int y = procWindow.y1; y < procWindow.y2; ++y) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);
            PIX *srcPix2 = (PIX *) _srcImg2->getPixelAddress(procWindow.x1, y);


            double sumLine[nComponents]; // partial sum to avoid underflows
            std::fill(sumLine, sumLine + nComponents, 0.);

            for (int x = procWindow.x1; x < procWindow.x2; ++x) {
                for (int c = 0; c < nComponents; ++c) {
                    double v = *dstPix;
                    double v2 = *srcPix2;
                    sumLine[c] += fabs(v-v2);
                    ++dstPix;
                    ++srcPix2;
                }
            }
            for (int c = 0; c < nComponents; ++c) {
                sum[c] += sumLine[c];
            }
            count += procWindow.x2 - procWindow.x1;
        }

        addResults(sum, count);
    }
};


////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class ImageStatisticsPlugin
    : public ImageEffect
{
public:
    /** @brief ctor */
    ImageStatisticsPlugin(OfxImageEffectHandle handle)
        : ImageEffect(handle)
        , _dstClip(0)
        , _srcClip(0)
    {
        _dstClip = fetchClip(kOfxImageEffectOutputClipName);
        assert( _dstClip && (!_dstClip->isConnected() || _dstClip->getPixelComponents() == ePixelComponentAlpha ||
                             _dstClip->getPixelComponents() == ePixelComponentRGB ||
                             _dstClip->getPixelComponents() == ePixelComponentRGBA) );
        _srcClip = getContext() == eContextGenerator ? NULL : fetchClip(kOfxImageEffectSimpleSourceClipName);
        assert( (!_srcClip && getContext() == eContextGenerator) ||
                ( _srcClip && (!_srcClip->isConnected() || _srcClip->getPixelComponents() ==  ePixelComponentAlpha ||
                               _srcClip->getPixelComponents() == ePixelComponentRGB ||
                               _srcClip->getPixelComponents() == ePixelComponentRGBA) ) );

        _SDA = fetchDoubleParam(kParamSDA);
        assert(_SDA);
        _analyzeSequence = fetchPushButtonParam(kParamAnalyzeSequence);
        assert(_analyzeSequence);
       
    }

private:
    /* override is identity */
    virtual bool isIdentity(const IsIdentityArguments &args, Clip * &identityClip, double &identityTime) OVERRIDE FINAL;


    /* Override the render */
    virtual void render(const RenderArguments &args) OVERRIDE FINAL;
    virtual void getRegionsOfInterest(const RegionsOfInterestArguments &args, RegionOfInterestSetter &rois) OVERRIDE FINAL;
    virtual bool getRegionOfDefinition(const RegionOfDefinitionArguments &args, OfxRectD & rod) OVERRIDE FINAL;
    virtual void changedParam(const InstanceChangedArgs &args, const std::string &paramName) OVERRIDE FINAL;
    virtual void getFramesNeeded(const OFX::FramesNeededArguments &args, OFX::FramesNeededSetter &frames);
    /* set up and run a processor */
    void setupAndProcess(ImageStatisticsProcessorBase &processor, const Image* srcImg, double time, const OfxRectI &analysisWindow, const Results &prevResults, Results *results);

    // compute computation window in srcImg
    bool computeWindow(const Image* srcImg, double time, OfxRectI *analysisWindow);

    // update image statistics
    void update(const Image* srcImg, double time, const OfxRectI& analysisWindow);


    template <template<class PIX, int nComponents, int maxValue> class Processor, class PIX, int nComponents, int maxValue>
    void updateSubComponentsDepth(const Image* srcImg,
                                  double time,
                                  const OfxRectI &analysisWindow,
                                  const Results& prevResults,
                                  Results* results)
    {
        Processor<PIX, nComponents, maxValue> fred(*this);
        setupAndProcess(fred, srcImg, time, analysisWindow, prevResults, results);
    }


    template <template<class PIX, int nComponents, int maxValue> class Processor, int nComponents>
    void updateSubComponents(const Image* srcImg,
                             double time,
                             const OfxRectI &analysisWindow,
                             const Results& prevResults,
                             Results* results)
    {
        BitDepthEnum srcBitDepth = srcImg->getPixelDepth();

        switch (srcBitDepth) {
        case eBitDepthUByte: {
            updateSubComponentsDepth<Processor, unsigned char, nComponents, 255>(srcImg, time, analysisWindow, prevResults, results);
            break;
        }
        case eBitDepthUShort: {
            updateSubComponentsDepth<Processor, unsigned short, nComponents, 65535>(srcImg, time, analysisWindow, prevResults, results);
            break;
        }
        case eBitDepthFloat: {
            updateSubComponentsDepth<Processor, float, nComponents, 1>(srcImg, time, analysisWindow, prevResults, results);
            break;
        }
        default:
            throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

    template <template<class PIX, int nComponents, int maxValue> class Processor>
    void updateSub(const Image* srcImg,
                   double time,
                   const OfxRectI &analysisWindow,
                   const Results& prevResults,
                   Results* results)
    {
        PixelComponentEnum srcComponents  = srcImg->getPixelComponents();

        assert(srcComponents == ePixelComponentAlpha || srcComponents == ePixelComponentRGB || srcComponents == ePixelComponentRGBA);
        if (srcComponents == ePixelComponentAlpha) {
            updateSubComponents<Processor, 1>(srcImg, time, analysisWindow, prevResults, results);
        } else if (srcComponents == ePixelComponentRGBA) {
            updateSubComponents<Processor, 4>(srcImg, time, analysisWindow, prevResults, results);
        } else if (srcComponents == ePixelComponentRGB) {
            updateSubComponents<Processor, 3>(srcImg, time, analysisWindow, prevResults, results);
        } else {
            // coverity[dead_error_line]
            throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

private:

    // do not need to delete these, the ImageEffect is managing them for us
    Clip *_dstClip;
    Clip *_srcClip;

    DoubleParam* _SDA;
    PushButtonParam* _analyzeSequence;

};

////////////////////////////////////////////////////////////////////////////////
/** @brief render for the filter */


// the overridden render function
void
ImageStatisticsPlugin::render(const RenderArguments &args)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    assert( kSupportsMultipleClipPARs   || !_srcClip || _srcClip->getPixelAspectRatio() == _dstClip->getPixelAspectRatio() );
    assert( kSupportsMultipleClipDepths || !_srcClip || _srcClip->getPixelDepth()       == _dstClip->getPixelDepth() );
    // do the rendering
    std::auto_ptr<Image> dst( _dstClip->fetchImage(args.time) );
    if ( !dst.get() ) {
        throwSuiteStatusException(kOfxStatFailed);
    }
    if ( (dst->getRenderScale().x != args.renderScale.x) ||
         ( dst->getRenderScale().y != args.renderScale.y) ||
         ( ( dst->getField() != eFieldNone) /* for DaVinci Resolve */ && ( dst->getField() != args.fieldToRender) ) ) {
        setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
        throwSuiteStatusException(kOfxStatFailed);
    }
    BitDepthEnum dstBitDepth       = dst->getPixelDepth();
    PixelComponentEnum dstComponents  = dst->getPixelComponents();
    std::auto_ptr<const Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                    _srcClip->fetchImage(args.time) : 0 );
    if ( src.get() ) {
        if ( (src->getRenderScale().x != args.renderScale.x) ||
             ( src->getRenderScale().y != args.renderScale.y) ||
             ( ( src->getField() != eFieldNone) /* for DaVinci Resolve */ && ( src->getField() != args.fieldToRender) ) ) {
            setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
            throwSuiteStatusException(kOfxStatFailed);
        }
        BitDepthEnum srcBitDepth      = src->getPixelDepth();
        PixelComponentEnum srcComponents = src->getPixelComponents();
        if ( (srcBitDepth != dstBitDepth) || (srcComponents != dstComponents) ) {
            throwSuiteStatusException(kOfxStatErrImageFormat);
        }
    }

    copyPixels( *this, args.renderWindow, src.get(), dst.get() );

} // ImageStatisticsPlugin::render


void
ImageStatisticsPlugin::getFramesNeeded(const OFX::FramesNeededArguments &args, OFX::FramesNeededSetter &frames) {
    const double time = args.time;
    OfxRangeD range;
    range.min = time;
    range.max = time + 1;
    frames.setFramesNeeded(*_srcClip, range);
}

// override the roi call
// Required if the plugin requires a region from the inputs which is different from the rendered region of the output.
// (this is the case here)
void
ImageStatisticsPlugin::getRegionsOfInterest(const RegionsOfInterestArguments &args,
                                            RegionOfInterestSetter &rois)
{}

bool
ImageStatisticsPlugin::getRegionOfDefinition(const RegionOfDefinitionArguments &args,
                                             OfxRectD & /*rod*/)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    return false;
}

bool
ImageStatisticsPlugin::isIdentity(const IsIdentityArguments &args,
                                  Clip * &identityClip,
                                  double & /*identityTime*/)
{

    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    const double time = args.time;
    
    identityClip = _srcClip;
    return true;

}

void
ImageStatisticsPlugin::changedParam(const InstanceChangedArgs &args,
                                    const std::string &paramName)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    bool doAnalyzeRGBA = false;
    bool doAnalyzeSequenceRGBA = false;
    OfxRectI analysisWindow;
    const double time = args.time;

    if (paramName == kParamAnalyzeSequence) {
        doAnalyzeSequenceRGBA = true;
    }

    if (paramName == kParamClearSequence) {
        _SDA->deleteAllKeys();
    }

    
    // RGBA analysis
    if ((doAnalyzeRGBA) && _srcClip && _srcClip->isConnected()) {
        std::auto_ptr<Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                  _srcClip->fetchImage(args.time) : 0 );
        if ( src.get() ) {
            if ( (src->getRenderScale().x != args.renderScale.x) ||
                 ( src->getRenderScale().y != args.renderScale.y) ) {
                setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
                throwSuiteStatusException(kOfxStatFailed);
            }
            bool intersect = computeWindow(src.get(), args.time, &analysisWindow);
            if (intersect) {
#             ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
                getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 1, false);
#             endif
                beginEditBlock("analyzeFrame");
                if (doAnalyzeRGBA) {
                    update(src.get(), args.time, analysisWindow);
                }
                
                endEditBlock();
#             ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
                getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 0, false);
#             endif
            }
        }
    }
    if ((doAnalyzeSequenceRGBA) && _srcClip && _srcClip->isConnected()) {
#     ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
        getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 1, false);
#     endif
        progressStart("Analyzing sequence...");
        beginEditBlock("analyzeSequence");
        OfxRangeD range = _srcClip->getFrameRange();
        //timeLineGetBounds(range.min, range.max); // wrong: we want the input frame range only
        int tmin = (int)std::ceil(range.min);
        int tmax = (int)std::floor(range.max);
        for (int t = tmin; t <= tmax; ++t) {
            std::auto_ptr<Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                      _srcClip->fetchImage(t) : 0 );
            if ( src.get() ) {
                if ( (src->getRenderScale().x != args.renderScale.x) ||
                     ( src->getRenderScale().y != args.renderScale.y) ) {
                    setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
                    throwSuiteStatusException(kOfxStatFailed);
                }
                bool intersect = computeWindow(src.get(), t, &analysisWindow);
                if (intersect) {
                    if (doAnalyzeSequenceRGBA) {
                        update(src.get(), t, analysisWindow);
                    }
                }
            }
            if (tmax != tmin) {
                if ( !progressUpdate( (t - tmin) / (double)(tmax - tmin) ) ) {
                    break;
                }
            }
        }
        progressEnd();
        endEditBlock();
#     ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
        getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 0, false);
#     endif
    }
} // ImageStatisticsPlugin::changedParam

/* set up and run a processor */
void
ImageStatisticsPlugin::setupAndProcess(ImageStatisticsProcessorBase &processor,
                                       const Image* srcImg,
                                       double time,
                                       const OfxRectI &analysisWindow,
                                       const Results &prevResults,
                                       Results *results)
{
    // set the images
    processor.setDstImg( const_cast<Image*>(srcImg) ); // not a bug: we only set dst

    std::auto_ptr<const OFX::Image> src2((_srcClip && _srcClip->isConnected()) ?
        _srcClip->fetchImage(time + 1) : 0);

    const OFX::Image *_srcImg2;
    processor.setSrcImg2(const_cast<OFX::Image*>(src2.get()));

    // set the render window
    processor.setRenderWindow(analysisWindow);

    processor.setPrevResults(prevResults);

    // Call the base class process member, this will call the derived templated process code
    processor.process();

    if ( !abort() ) {
        processor.getResults(results);
    }
}

bool
ImageStatisticsPlugin::computeWindow(const Image* srcImg,
                                     double time,
                                     OfxRectI *analysisWindow)
{
    OfxRectD regionOfInterest;
    
    if (_srcClip) {
        // use the src region of definition as rectangle, but avoid infinite rectangle
        regionOfInterest = _srcClip->getRegionOfDefinition(time);
        OfxPointD size = getProjectSize();
        OfxPointD offset = getProjectOffset();
        if (regionOfInterest.x1 <= kOfxFlagInfiniteMin) {
            regionOfInterest.x1 = offset.x;
        }
        if (regionOfInterest.x2 >= kOfxFlagInfiniteMax) {
            regionOfInterest.x2 = offset.x + size.x;
        }
        if (regionOfInterest.y1 <= kOfxFlagInfiniteMin) {
            regionOfInterest.y1 = offset.y;
        }
        if (regionOfInterest.y2 >= kOfxFlagInfiniteMax) {
            regionOfInterest.y2 = offset.y + size.y;
        }
    } else {
        regionOfInterest.x2 += regionOfInterest.x1;
        regionOfInterest.y2 += regionOfInterest.y1;
    }
    Coords::toPixelEnclosing(regionOfInterest,
                             srcImg->getRenderScale(),
                             srcImg->getPixelAspectRatio(),
                             analysisWindow);

    return Coords::rectIntersection(*analysisWindow, srcImg->getBounds(), analysisWindow);
}

// update image statistics
void
ImageStatisticsPlugin::update(const Image* srcImg,
                              double time,
                              const OfxRectI &analysisWindow)
{
    // TODO: CHECK if checkDoubleAnalysis param is true and analysisWindow is the same as btmLeft/sizeAnalysis
    Results results;

    if ( !abort() ) {
        updateSub<ImageMeanProcessor>(srcImg, time, analysisWindow, results, &results);
    }
    if ( abort() ) {
        return;
    }

    _SDA->setValueAtTime(time, results.mean.r);


}

mDeclarePluginFactory(ImageStatisticsPluginFactory, {}, {});

void
ImageStatisticsPluginFactory::describe(ImageEffectDescriptor &desc)
{
    // basic labels
    desc.setLabel(kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

    desc.addSupportedContext(eContextGeneral);
    desc.addSupportedContext(eContextFilter);

    desc.addSupportedBitDepth(eBitDepthUByte);
    desc.addSupportedBitDepth(eBitDepthUShort);
    desc.addSupportedBitDepth(eBitDepthFloat);


    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(true);
    desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);
    desc.setSupportsMultipleClipDepths(kSupportsMultipleClipDepths);
    desc.setRenderThreadSafety(kRenderThreadSafety);

    desc.setSupportsTiles(kSupportsTiles);

    // in order to support multiresolution, render() must take into account the pixelaspectratio and the renderscale
    // and scale the transform appropriately.
    // All other functions are usually in canonical coordinates.
    desc.setSupportsMultiResolution(kSupportsMultiResolution);
    //desc.setOverlayInteractDescriptor(new ImageStatisticsOverlayDescriptor);
#ifdef OFX_EXTENSIONS_NATRON
    desc.setChannelSelector(ePixelComponentNone);
#endif
}

ImageEffect*
ImageStatisticsPluginFactory::createInstance(OfxImageEffectHandle handle,
                                             ContextEnum /*context*/)
{
    return new ImageStatisticsPlugin(handle);
}

void
ImageStatisticsPluginFactory::describeInContext(ImageEffectDescriptor &desc,
                                                ContextEnum /*context*/)
{
    // Source clip only in the filter context
    // create the mandated source clip
    // always declare the source clip first, because some hosts may consider
    // it as the default input clip (e.g. Nuke)
    ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);

    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->addSupportedComponent(ePixelComponentRGB);
    srcClip->addSupportedComponent(ePixelComponentAlpha);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);
    srcClip->setOptional(false);

    // create the mandated output clip
    ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentRGB);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // make some pages and to things in
    PageParamDescriptor *page = desc.definePageParam("Controls");

    
    {
        
        // SDA
        {
            DoubleParamDescriptor* param = desc.defineDoubleParam(kParamSDA);
            param->setLabel(kParamSDALabel);
            param->setHint(kParamSDAHint);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (page) {
                page->addChild(*param);
            }
        }

        {
            DoubleParamDescriptor* param = desc.defineDoubleParam(kParamThreshold);
            param->setLabel(kParamThresholdLabel);
            param->setHint(kParamThresholdHint);
            param->setRange(0., 1.); // Resolve requires range and display range or values are clamped to (-1,1)
            param->setDisplayRange(0., 1.);
            param->setDefault(1.);
            if (page) {
                page->addChild(*param);
            }
        }

        // analyzeSequence
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamAnalyzeSequence);
            param->setLabel(kParamAnalyzeSequenceLabel);
            param->setHint(kParamAnalyzeSequenceHint);
            if (page) {
                page->addChild(*param);
            }
        }

        // clearSequence
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamClearSequence);
            param->setLabel(kParamClearSequenceLabel);
            param->setHint(kParamClearSequenceHint);
            if (page) {
                page->addChild(*param);
            }
        }
    }

    
} // ImageStatisticsPluginFactory::describeInContext

static ImageStatisticsPluginFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
mRegisterPluginFactoryInstance(p)

OFXS_NAMESPACE_ANONYMOUS_EXIT
