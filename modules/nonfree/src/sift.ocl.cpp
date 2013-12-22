/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// Authors:
//  * Peter Andreas Entschev, peter@entschev.com
//
//M*/

#include "precomp.hpp"
#include "opencl_kernels.hpp"

// TODO: REMOVE
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace cv::ocl;

namespace
{
    /******************************* Defs and macros *****************************/

    // default number of sampled intervals per octave
    static const int SIFT_INTVLS = 3;

    // default sigma for initial gaussian smoothing
    static const float SIFT_SIGMA = 1.6f;

    // default threshold on keypoint contrast |D(x)|
    static const float SIFT_CONTR_THR = 0.04f;

    // default threshold on keypoint ratio of principle curvatures
    static const float SIFT_CURV_THR = 10.f;

    // double image size before pyramid construction?
    static const bool SIFT_IMG_DBL = true;

    // default width of descriptor histogram array
    static const int SIFT_DESCR_WIDTH = 4;

    // default number of bins per histogram in descriptor array
    static const int SIFT_DESCR_HIST_BINS = 8;

    // assumed gaussian blur for input image
    static const float SIFT_INIT_SIGMA = 0.5f;

    // width of border in which to ignore keypoints
    static const int SIFT_IMG_BORDER = 5;

    // maximum steps of keypoint interpolation before failure
    static const int SIFT_MAX_INTERP_STEPS = 5;

    // default number of bins in histogram for orientation assignment
    static const int SIFT_ORI_HIST_BINS = 36;

    // determines gaussian sigma for orientation assignment
    static const float SIFT_ORI_SIG_FCTR = 1.5f;

    // determines the radius of the region used in orientation assignment
    static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

    // orientation magnitude relative to max that results in new feature
    static const float SIFT_ORI_PEAK_RATIO = 0.8f;

    // determines the size of a single descriptor orientation histogram
    static const float SIFT_DESCR_SCL_FCTR = 3.f;

    // threshold on magnitude of elements of descriptor vector
    static const float SIFT_DESCR_MAG_THR = 0.2f;

    // factor used to convert floating-point descriptor to unsigned char
    static const float SIFT_INT_DESCR_FCTR = 512.f;

    #if 0
    // intermediate type used for DoG pyramids
    typedef short sift_wt;
    static const int SIFT_FIXPT_SCALE = 48;
    #else
    // intermediate type used for DoG pyramids
    typedef float sift_wt;
    static const int SIFT_FIXPT_SCALE = 1;
    #endif
}


static oclMat createInitialImage(const oclMat& img, bool doubleImageSize, float sigma)
{
    oclMat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        img.copyTo(gray);
    gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

    float sig_diff;

    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        oclMat dbl;
        resize(gray_fpt, dbl, Size(gray.cols*2, gray.rows*2), 0, 0, INTER_LINEAR);
        GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
        return dbl;
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
        return gray_fpt;
    }
}

void SIFT_OCL::buildGaussianPyramid(const oclMat& base, std::vector<oclMat>& pyr, int nOctaves) const
{
    std::vector<double> sig(nOctaveLayers + 3);
    pyr.resize(nOctaves*(nOctaveLayers + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = std::pow( 2., 1. / nOctaveLayers );
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = std::pow(k, (double)(i-1))*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 3; i++ )
        {
            oclMat& dst = pyr[o*(nOctaveLayers + 3) + i];
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                const oclMat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
                resize(src, dst, Size(src.cols/2, src.rows/2),
                       0, 0, INTER_NEAREST);
            }
            else
            {
                const oclMat& src = pyr[o*(nOctaveLayers + 3) + i-1];
                GaussianBlur(src, dst, Size(), sig[i], sig[i]);
            }
        }
    }
}

void SIFT_OCL::buildDoGPyramid( const std::vector<oclMat>& gpyr, std::vector<oclMat>& dogpyr ) const
{
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
    dogpyr.resize( nOctaves*(nOctaveLayers + 2) );

    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 2; i++ )
        {
            const oclMat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
            const oclMat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
            oclMat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
            subtract(src2, src1, dst, oclMat());
        }
    }
}

static void findExtrema_OCL(const oclMat& prev, const oclMat& center, const oclMat& next,
                            oclMat& keypoints)
{
    CV_Assert(prev.rows == center.rows && center.rows == next.rows &&
              prev.cols == center.cols && center.cols == next.cols &&
              prev.depth() == center.depth() && center.depth() == next.depth() &&
              prev.channels() == center.channels() && center.channels() == next.channels());

    int cn = prev.channels();
    int depth = prev.depth();

    size_t localThreads[3] = {32, 8, 1};
    size_t globalThreads[3] = {divUp(center.rows, localThreads[1]) * localThreads[1] * localThreads[0],
                               1,
                               1};

    const char * const channelMap[] = { "", "", "2", "4", "4" };
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    std::string buildOptions = format("-D T=%s%s -D %s", typeMap[depth], channelMap[cn]);

    Context *clCxt = Context::getContext();
    String kernelName = "HarrisResponses";
    std::vector< std::pair<size_t, const void *> > args;

    int imgStep = img.step / img.elemSize();
    int keypointsStep = keypoints.step / keypoints.elemSize();

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&img.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&npoints));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&blockSize));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&harris_k));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&imgStep));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsStep));

    openCLExecuteKernel(clCxt, &orb, kernelName, globalThreads, localThreads, args, -1, -1, (char*)"-D CPU");

    bool is_cpu = isCpuDevice();
    if (is_cpu)
        openCLExecuteKernel(clCxt, &orb, kernelName, globalThreads, localThreads, args, -1, -1, (char*)"-D CPU");
    else
    {
        cl_kernel kernel = openCLGetKernelFromSource(Context::getContext(), &orb, kernelName);
        int wave_size = (int)queryWaveFrontSize(kernel);
        openCLSafeCall(clReleaseKernel(kernel));

        std::string opt = format("-D WAVE_SIZE=%d", wave_size);
        openCLExecuteKernel(Context::getContext(), &orb, kernelName, globalThreads, localThreads, args, -1, -1, opt.c_str());
    }
}

//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
//void SIFT_OCL::findScaleSpaceExtrema(const std::vector<oclMat>& gauss_pyr, const std::vector<oclMat>& dog_pyr,
//                                     std::vector<KeyPoint>& keypoints) const
void SIFT_OCL::findScaleSpaceExtrema(const std::vector<oclMat>& gauss_pyr, const std::vector<oclMat>& dog_pyr,
                                     oclMat& keypoints) const
{
    int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers + 3);
    int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
    const int n = SIFT_ORI_HIST_BINS;
    float hist[n];
//    KeyPoint kpt;
//
//    keypoints.clear();

    maxFeatures = std::min(static_cast<int>(img.size().area() * keypointsRatio_), 65535);
    ensureSizeIsEnough(ROWS_COUNT, maxFeatures, CV_32FC1, keypoints);

    for( int o = 0; o < nOctaves; o++ )
        for( int i = 1; i <= nOctaveLayers; i++ )
        {
            int idx = o*(nOctaveLayers+2)+i;
            const oclMat& img = dog_pyr[idx];
            const oclMat& prev = dog_pyr[idx-1];
            const oclMat& next = dog_pyr[idx+1];
            int step = (int)img.step1();
            int rows = img.rows, cols = img.cols;

            findExtrema_OCL(prev, img, next, keypoints);

//            for( int r = SIFT_IMG_BORDER; r < rows-SIFT_IMG_BORDER; r++)
//            {
//                const sift_wt* currptr = img.ptr<sift_wt>(r);
//                const sift_wt* prevptr = prev.ptr<sift_wt>(r);
//                const sift_wt* nextptr = next.ptr<sift_wt>(r);
//
//                for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++)
//                {
//                    sift_wt val = currptr[c];
//
//                    // find local extrema with pixel accuracy
//                    if( std::abs(val) > threshold &&
//                       ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
//                         val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
//                         val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
//                         val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
//                         val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
//                         val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
//                         val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
//                         val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
//                         val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
//                        (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
//                         val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
//                         val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
//                         val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
//                         val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
//                         val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
//                         val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
//                         val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
//                         val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
//                    {
//                        int r1 = r, c1 = c, layer = i;
//                        if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
//                                                nOctaveLayers, (float)contrastThreshold,
//                                                (float)edgeThreshold, (float)sigma) )
//                            continue;
//                        float scl_octv = kpt.size*0.5f/(1 << o);
//                        float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
//                                                         Point(c1, r1),
//                                                         cvRound(SIFT_ORI_RADIUS * scl_octv),
//                                                         SIFT_ORI_SIG_FCTR * scl_octv,
//                                                         hist, n);
//                        float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
//                        for( int j = 0; j < n; j++ )
//                        {
//                            int l = j > 0 ? j - 1 : n - 1;
//                            int r2 = j < n-1 ? j + 1 : 0;
//
//                            if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
//                            {
//                                float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
//                                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
//                                kpt.angle = 360.f - (float)((360.f/n) * bin);
//                                if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
//                                    kpt.angle = 0.f;
//                                keypoints.push_back(kpt);
//                            }
//                        }
//                    }
//                }
//            }
        }
}


cv::ocl::SIFT_OCL::SIFT_OCL(int _nfeatures, int _nOctaveLayers,
                            double _contrastThreshold, double _edgeThreshold, double _sigma ) :
                            nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
                            contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
{
}

int cv::ocl::SIFT_OCL::descriptorSize() const
{
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int cv::ocl::SIFT_OCL::descriptorType() const
{
    return CV_32F;
}

int cv::ocl::SIFT_OCL::defaultNorm() const
{
    return NORM_L2;
}

void cv::ocl::SIFT_OCL::operator()(oclMat& image, oclMat& mask,
                                   std::vector<KeyPoint>& keypoints) const
{
    //(*this)(image, mask, keypoints, oclMat());

    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( !mask.empty() && mask.type() != CV_8UC1 )
        CV_Error( Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

//    if( useProvidedKeypoints )
//    {
//        firstOctave = 0;
//        int maxOctave = INT_MIN;
//        for( size_t i = 0; i < keypoints.size(); i++ )
//        {
//            int octave, layer;
//            float scale;
//            unpackOctave(keypoints[i], octave, layer, scale);
//            firstOctave = std::min(firstOctave, octave);
//            maxOctave = std::max(maxOctave, octave);
//            actualNLayers = std::max(actualNLayers, layer-2);
//        }
//
//        firstOctave = std::min(firstOctave, 0);
//        CV_Assert( firstOctave >= -1 && actualNLayers <= nOctaveLayers );
//        actualNOctaves = maxOctave - firstOctave + 1;
//    }

    oclMat base = createInitialImage(image, firstOctave < 0, (float)sigma);
    std::vector<oclMat> gpyr, dogpyr;
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log( (double)std::min( base.cols, base.rows ) ) / std::log(2.) - 2) - firstOctave;

//    Mat firstlevel(base);
//    firstlevel.convertTo(firstlevel, CV_8U);
//    imshow("firstlevel", firstlevel);
//    waitKey(0);

    //double t, tf = getTickFrequency();
    //t = (double)getTickCount();
    buildGaussianPyramid(base, gpyr, nOctaves);
    buildDoGPyramid(gpyr, dogpyr);

    //t = (double)getTickCount() - t;
    //printf("pyramid construction time: %g\n", t*1000./tf);

    if( !useProvidedKeypoints )
    {
        //t = (double)getTickCount();
        findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
//        KeyPointsFilter::removeDuplicated( keypoints );
//
//        if( nfeatures > 0 )
//            KeyPointsFilter::retainBest(keypoints, nfeatures);
//        //t = (double)getTickCount() - t;
//        //printf("keypoint detection time: %g\n", t*1000./tf);
//
//        if( firstOctave < 0 )
//            for( size_t i = 0; i < keypoints.size(); i++ )
//            {
//                KeyPoint& kpt = keypoints[i];
//                float scale = 1.f/(float)(1 << -firstOctave);
//                kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
//                kpt.pt *= scale;
//                kpt.size *= scale;
//            }
//
//        if( !mask.empty() )
//            KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }
//    else
//    {
//        // filter keypoints by mask
//        //KeyPointsFilter::runByPixelsMask( keypoints, mask );
//    }
//
//    if( _descriptors.needed() )
//    {
//        //t = (double)getTickCount();
//        int dsize = descriptorSize();
//        _descriptors.create((int)keypoints.size(), dsize, CV_32F);
//        Mat descriptors = _descriptors.getMat();
//
//        calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
//        //t = (double)getTickCount() - t;
//        //printf("descriptor extraction time: %g\n", t*1000./tf);
//    }
}

//void cv::ocl::ORB_OCL::downloadKeyPoints(const oclMat &d_keypoints, std::vector<KeyPoint>& keypoints)
//{
//    if (d_keypoints.empty())
//    {
//        keypoints.clear();
//        return;
//    }
//
//    Mat h_keypoints(d_keypoints);
//
//    convertKeyPoints(h_keypoints, keypoints);
//}
//
//void cv::ocl::ORB_OCL::convertKeyPoints(const Mat &d_keypoints, std::vector<KeyPoint>& keypoints)
//{
//    if (d_keypoints.empty())
//    {
//        keypoints.clear();
//        return;
//    }
//
//    CV_Assert(d_keypoints.type() == CV_32FC1 && d_keypoints.rows == ROWS_COUNT);
//
//    const float* x_ptr = d_keypoints.ptr<float>(X_ROW);
//    const float* y_ptr = d_keypoints.ptr<float>(Y_ROW);
//    const float* response_ptr = d_keypoints.ptr<float>(RESPONSE_ROW);
//    const float* angle_ptr = d_keypoints.ptr<float>(ANGLE_ROW);
//    const float* octave_ptr = d_keypoints.ptr<float>(OCTAVE_ROW);
//    const float* size_ptr = d_keypoints.ptr<float>(SIZE_ROW);
//
//    keypoints.resize(d_keypoints.cols);
//
//    for (int i = 0; i < d_keypoints.cols; ++i)
//    {
//        KeyPoint kp;
//
//        kp.pt.x = x_ptr[i];
//        kp.pt.y = y_ptr[i];
//        kp.response = response_ptr[i];
//        kp.angle = angle_ptr[i];
//        kp.octave = static_cast<int>(octave_ptr[i]);
//        kp.size = size_ptr[i];
//
//        keypoints[i] = kp;
//    }
//}
//
//void cv::ocl::ORB_OCL::operator()(const oclMat& image, const oclMat& mask, oclMat& keypoints)
//{
//    buildScalePyramids(image, mask);
//    computeKeyPointsPyramid();
//    mergeKeyPoints(keypoints);
//}
//
//void cv::ocl::ORB_OCL::operator()(const oclMat& image, const oclMat& mask, oclMat& keypoints, oclMat& descriptors)
//{
//    buildScalePyramids(image, mask);
//    computeKeyPointsPyramid();
//    computeDescriptors(descriptors);
//    mergeKeyPoints(keypoints);
//}
//
//void cv::ocl::ORB_OCL::operator()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints)
//{
//    (*this)(image, mask, d_keypoints_);
//    downloadKeyPoints(d_keypoints_, keypoints);
//}
//
//void cv::ocl::ORB_OCL::operator()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints, oclMat& descriptors)
//{
//    (*this)(image, mask, d_keypoints_, descriptors);
//    downloadKeyPoints(d_keypoints_, keypoints);
//}
//
//void cv::ocl::ORB_OCL::release()
//{
//    imagePyr_.clear();
//    maskPyr_.clear();
//
//    buf_.release();
//
//    keyPointsPyr_.clear();
//
//    fastDetector_.release();
//
//    d_keypoints_.release();
//
//    uMax_.release();
//}
