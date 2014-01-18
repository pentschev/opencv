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

/**********************************************************************************************\
 Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/sift/
 Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

#include "precomp.hpp"
#include "opencl_kernels.hpp"

// TODO: REMOVE
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ocl;

static ProgramEntry siftprog = cv::ocl::nonfree::sift;
static ProgramEntry keypoint_sort_prog = cv::ocl::nonfree::keypoint_sort;

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

//////////////////////////////////////// Static Methods ////////////////////////////////////////

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


/////////////////////////////// Static methods to call SIFT OpenCL kernels ///////////////////////////////

static void calcOrientationHist_OCL(const std::vector<oclMat>& gausspyr, const oclMat& keypointsIn,
                                    oclMat& keypointsOut, const int octaveKeypoints, int& counter,
                                    const int octave, const int nOctaveLayers, const int firstOctave)
{
    CV_Assert(nOctaveLayers <= 3);

    int octaveIdx = octave*(nOctaveLayers+3);

    int cn = gausspyr[octaveIdx].channels();
    int depth = gausspyr[octaveIdx].depth();
    int rows = gausspyr[octaveIdx].rows;
    int cols = gausspyr[octaveIdx].cols;

//    size_t localThreads[3] = {1, 256, 1};
//    size_t localThreads[3] = {16, 16, 1};
    size_t localThreads[3] = {4, 64, 1};
    size_t globalThreads[3] = {divUp(octaveKeypoints, localThreads[0]) * localThreads[0],
//    size_t globalThreads[3] = {(octaveKeypoints / localThreads[0]) * localThreads[0],
                               1,
                               1};

    const char * const channelMap[] = { "", "", "2", "4", "4" };
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    String buildOptions = format("-D T=%s%s", typeMap[depth], channelMap[cn]);

    Context *clCxt = Context::getContext();
    String kernelName = "calcOrientationHist";
    std::vector< std::pair<size_t, const void *> > args;

    int layerStep[3] = {0,0,0};
    for (int i = 1; i <= nOctaveLayers; i++)
        layerStep[i-1] = gausspyr[octaveIdx+i].step / gausspyr[octaveIdx+i].elemSize();

    int keypointsInStep = keypointsIn.step / keypointsIn.elemSize();
    int keypointsOutStep = keypointsOut.step / keypointsOut.elemSize();

    std::cout << "octaveKeypoints: " << octaveKeypoints << std::endl;

    int err = CL_SUCCESS;
    cl_mem counterCL = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(),
                                    CL_MEM_COPY_HOST_PTR, sizeof(int),
                                    &counter, &err);

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&gausspyr[octaveIdx+1].data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[0]));
    if (nOctaveLayers >= 2)
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&gausspyr[octaveIdx+2].data));
    else
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)NULL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[1]));
    if (nOctaveLayers == 3)
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&gausspyr[octaveIdx+3].data));
    else
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)NULL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[2]));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypointsIn.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsInStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypointsOut.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsOutStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&counterCL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&octaveKeypoints));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&firstOctave));

    openCLExecuteKernel(clCxt, &siftprog, kernelName, globalThreads, localThreads, args, -1, -1, buildOptions.c_str());

    clEnqueueReadBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(),
                        counterCL, CL_TRUE, 0, sizeof(int), &counter, 0, NULL, NULL);
    clReleaseMemObject(counterCL);
    std::cout << "orientation counter: " << counter << std::endl;

    keypointsOut.cols = counter;
}


static void adjustLocalExtrema_OCL(const std::vector<oclMat>& dogpyr, const oclMat& keypointsIn,
                                   oclMat& keypointsOut, const int octaveKeypoints,
                                   const int maxKeypoints, int& counter, const int octave,
                                   const int nOctaveLayers, const float contrastThreshold,
                                   const float edgeThreshold, const float sigma)
{
    CV_Assert(nOctaveLayers <= 3);

    int octaveIdx = octave*(nOctaveLayers+2);

    int cn = dogpyr[octaveIdx].channels();
    int depth = dogpyr[octaveIdx].depth();
    int rows = dogpyr[octaveIdx].rows;
    int cols = dogpyr[octaveIdx].cols;

    size_t localThreads[3] = {256, 1, 1};
    size_t globalThreads[3] = {divUp(octaveKeypoints, localThreads[0]) * localThreads[0],
                               1,
                               1};

    const char * const channelMap[] = { "", "", "2", "4", "4" };
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    String buildOptions = format("-D T=%s%s", typeMap[depth], channelMap[cn]);

    Context *clCxt = Context::getContext();
    String kernelName = "adjustLocalExtrema";
    std::vector< std::pair<size_t, const void *> > args;

    int layerStep[5] = {0,0,0,0,0};
    for (int i = 0; i < nOctaveLayers+2; i++)
        layerStep[i] = dogpyr[octaveIdx+i].step / dogpyr[octaveIdx+i].elemSize();

    int keypointsInStep = keypointsIn.step / keypointsIn.elemSize();
    int keypointsOutStep = keypointsOut.step / keypointsOut.elemSize();
    std::cout << "In: " << keypointsIn.cols << " " << keypointsIn.rows << " " << keypointsInStep << std::endl;
    std::cout << "Out: " << keypointsOut.cols << " " << keypointsOut.rows << " " << keypointsOutStep << std::endl;

    int err = CL_SUCCESS;
    cl_mem counterCL = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(),
                                    CL_MEM_COPY_HOST_PTR, sizeof(int),
                                    &counter, &err);

    std::cout << "counter1: " << counter << std::endl;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dogpyr[octaveIdx+0].data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[0]));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dogpyr[octaveIdx+1].data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[1]));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dogpyr[octaveIdx+2].data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[2]));
    if (nOctaveLayers >= 2)
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dogpyr[octaveIdx+3].data));
    else
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)NULL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[3]));
    if (nOctaveLayers == 3)
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dogpyr[octaveIdx+4].data));
    else
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)NULL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[4]));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypointsIn.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsInStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypointsOut.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsOutStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&counterCL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&octaveKeypoints));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&maxKeypoints));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&octave));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&nOctaveLayers));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&contrastThreshold));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&edgeThreshold));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&sigma));

    openCLExecuteKernel(clCxt, &siftprog, kernelName, globalThreads, localThreads, args, -1, -1, buildOptions.c_str());

//    Mat b(keypointsOut);
//    std::cout << b.rows << " " << b.cols << std::endl;
//    std::cout << b.at<float>(0,0) << " " << b.at<float>(1,0) << " " << b.at<float>(2,0) << " " <<
//                 b.at<float>(3,0) << " " << b.at<float>(4,0) << " " << b.at<float>(5,0) << " " <<
//                 b.at<float>(6,0) << std::endl;

    clEnqueueReadBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(),
                        counterCL, CL_TRUE, 0, sizeof(int), &counter, 0, NULL, NULL);
    clReleaseMemObject(counterCL);
    std::cout << "adjust counter: " << counter << std::endl;
}

static void findExtrema_OCL(const oclMat& prev, const oclMat& center, const oclMat& next,
                            oclMat& keypoints, int& counter, const int octave, const int scale,
                            const int maxKeypoints, const int threshold, const int nOctaveLayers,
                            const float contrastThreshold, const float edgeThreshold,
                            const float sigma)
{
    CV_Assert(prev.rows == center.rows && center.rows == next.rows &&
              prev.cols == center.cols && center.cols == next.cols &&
              prev.depth() == center.depth() && center.depth() == next.depth() &&
              prev.channels() == center.channels() && center.channels() == next.channels());

    int cn = prev.channels();
    int depth = prev.depth();

    size_t localThreads[3] = {32, 8, 1};
    size_t globalThreads[3] = {divUp(center.cols, localThreads[0]) * localThreads[0],
                               divUp(center.rows, localThreads[1]) * localThreads[1],
                               1};

    const char * const channelMap[] = { "", "", "2", "4", "4" };
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    String buildOptions = format("-D T=%s%s", typeMap[depth], channelMap[cn]);
    //sprintf(optBufPtr, "-D WAVE_SIZE=%d", static_cast<int>(wave_size));

    Context *clCxt = Context::getContext();
    String kernelName = "findExtrema";
    std::vector< std::pair<size_t, const void *> > args;

    int prevStep = prev.step / prev.elemSize();
    int centerStep = center.step / center.elemSize();
    int nextStep = next.step / next.elemSize();
    int keypointsStep = keypoints.step / keypoints.elemSize();

    int err = CL_SUCCESS;
    cl_mem counterCL = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(),
                                      CL_MEM_COPY_HOST_PTR, sizeof(int),
                                      &counter, &err);

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&prev.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&prevStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&center.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&centerStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&next.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&nextStep));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&center.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&center.cols));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&counterCL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&octave));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&scale));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&maxKeypoints));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&threshold));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&nOctaveLayers));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&contrastThreshold));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&edgeThreshold));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&sigma));

    openCLExecuteKernel(clCxt, &siftprog, kernelName, globalThreads, localThreads, args, -1, -1,
                        buildOptions.c_str());

/*    openCLSafeCall(clEnqueueReadBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(),
                                       counterCL, CL_TRUE, 0, sizeof(int), &counter, 0, NULL, NULL));
    openCLSafeCall(clReleaseMemObject(counterCL));*/
    clEnqueueReadBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(),
                        counterCL, CL_TRUE, 0, sizeof(int), &counter, 0, NULL, NULL);
    clReleaseMemObject(counterCL);
}

static void calcDescriptors_OCL(const std::vector<oclMat>& gausspyr, const oclMat& keypoints,
                                oclMat& descriptors, const int pyrOctave, const int nOctaveLayers,
                                const float scale, const int keypointOffset, const int totalKeypoints,
                                const int d, const int n)
{
    CV_Assert(nOctaveLayers <= 3);

    int octaveIdx = pyrOctave*(nOctaveLayers+3);

    int cn = gausspyr[octaveIdx].channels();
    int depth = gausspyr[octaveIdx].depth();
    int rows = gausspyr[octaveIdx].rows;
    int cols = gausspyr[octaveIdx].cols;

    size_t localThreads[3] = {1, 256, 1};
    size_t globalThreads[3] = {divUp(totalKeypoints, localThreads[0]) * localThreads[0],
                               1,
                               1};

    const char * const channelMap[] = { "", "", "2", "4", "4" };
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    String buildOptions = format("-D T=%s%s", typeMap[depth], channelMap[cn]);

    Context *clCxt = Context::getContext();
    String kernelName = "calcSIFTDescriptor";
    std::vector< std::pair<size_t, const void *> > args;

    int layerStep[3] = {0,0,0};
    for (int i = 1; i <= nOctaveLayers; i++)
        layerStep[i-1] = gausspyr[octaveIdx+i].step / gausspyr[octaveIdx+i].elemSize();

    int keypointsStep = keypoints.step / keypoints.elemSize();
    int descriptorsStep = descriptors.step / descriptors.elemSize();

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&gausspyr[octaveIdx+1].data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[0]));
    if (nOctaveLayers >= 2)
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&gausspyr[octaveIdx+2].data));
    else
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)NULL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[1]));
    if (nOctaveLayers == 3)
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&gausspyr[octaveIdx+3].data));
    else
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)NULL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&layerStep[2]));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&descriptors.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&descriptorsStep));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&nOctaveLayers));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointOffset));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&totalKeypoints));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&d));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&n));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&scale));

    openCLExecuteKernel(clCxt, &siftprog, kernelName, globalThreads, localThreads, args, -1, -1,
                        buildOptions.c_str());
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

namespace
{

const char * depth_strings[] =
{
    "uchar",   //CV_8U
    "char",    //CV_8S
    "ushort",  //CV_16U
    "short",   //CV_16S
    "int",     //CV_32S
    "float",   //CV_32F
    "double"   //CV_64F
};

void static genSortBuildOption(const oclMat& keypoints, bool isGreaterThan, char * build_opt_buf)
{
    sprintf(build_opt_buf, "-D IS_GT=%d -D T=%s -D MAX_ELEMENTS=%d",
            isGreaterThan?1:0, depth_strings[keypoints.depth()], keypoints.rows);
}

inline bool isSizePowerOf2(size_t size)
{
    return ((size - 1) & (size)) == 0;
}

}

static void sortKeypoints_OCL(oclMat& keypoints, const oclMat& keyOrder, bool isGreaterThan)
{
    // Current implementation is limited to 10 key/value elements to prevent device memory leakage
    CV_Assert(keypoints.rows <= 10);

    Context * cxt = Context::getContext();

    const size_t GROUP_SIZE = cxt->getDeviceInfo().maxWorkGroupSize >= 256 ? 256: 128;

    size_t vecSize = static_cast<size_t>(keypoints.cols);

    size_t globalThreads[3] = {vecSize, 1, 1};
    size_t localThreads[3]  = {GROUP_SIZE, 1, 1};

    std::vector< std::pair<size_t, const void *> > args;
    char build_opt_buf [100];
    genSortBuildOption(keypoints, isGreaterThan, build_opt_buf);

    int keypointsStep = keypoints.step / keypoints.elemSize();

    String kernelname[] = {String("blockInsertionSort"), String("merge")};
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&keypoints.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&keyOrder.data));
    args.push_back(std::make_pair(sizeof(cl_uint), (void *)&vecSize));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&keyOrder.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&keypoints.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&keypointsStep));

    clock_t time;
    time = clock();
    openCLExecuteKernel(cxt, &keypoint_sort_prog, kernelname[0], globalThreads, localThreads, args,
                        -1, -1, build_opt_buf);
    std::cout << "TIME: " << (clock() - time) / (double)CLOCKS_PER_SEC << std::endl;

    time = clock();
    //  Early exit for the case of no merge passes, values are already in destination vector
    if(vecSize <= GROUP_SIZE)
    {
        return;
    }

    //  An odd number of elements requires an extra merge pass to sort
    size_t numMerges = 0;
    //  Calculate the log2 of vecSize, taking into acvecSize our block size from kernel 1 is 64
    //  this is how many merge passes we want
    size_t log2BlockSize = vecSize >> 6;
    for( ; log2BlockSize > 1; log2BlockSize >>= 1 )
    {
        ++numMerges;
    }
    //  Check to see if the input vector size is a power of 2, if not we will need last merge pass
    numMerges += isSizePowerOf2(vecSize)? 1: 0;

    //  Allocate a flipflop buffer because the merge passes are out of place
    oclMat tmpKeypointBuffer(keypoints.size(), keypoints.type());
    args.resize(9);

    int tmpKeypointBufferStep = tmpKeypointBuffer.step / tmpKeypointBuffer.elemSize();

    args[2] = std::make_pair(sizeof(cl_mem), (void *)&keyOrder.data);
    args[3] = std::make_pair(sizeof(cl_uint), (void *)&vecSize);
    args[5] = std::make_pair(sizeof(cl_int), (void *)&keyOrder.cols);
    args[6] = std::make_pair(sizeof(cl_int), (void *)&keypoints.rows);


    for(size_t pass = 1; pass <= numMerges; ++pass )
    {
        //  For each pass, flip the input-output buffers
        if( pass & 0x1 )
        {
            args[0] = std::make_pair(sizeof(cl_mem), (void *)&keypoints.data);
            args[1] = std::make_pair(sizeof(cl_mem), (void *)&tmpKeypointBuffer.data);
            args[7] = std::make_pair(sizeof(cl_int), (void *)&keypointsStep);
            args[8] = std::make_pair(sizeof(cl_int), (void *)&tmpKeypointBufferStep);
        }
        else
        {
            args[0] = std::make_pair(sizeof(cl_mem), (void *)&tmpKeypointBuffer.data);
            args[1] = std::make_pair(sizeof(cl_mem), (void *)&keypoints.data);
            args[7] = std::make_pair(sizeof(cl_int), (void *)&tmpKeypointBufferStep);
            args[8] = std::make_pair(sizeof(cl_int), (void *)&keypointsStep);
        }
        //  For each pass, the merge window doubles
        unsigned int srcLogicalBlockSize = static_cast<unsigned int>( localThreads[0] << (pass-1) );
        args[4] = std::make_pair(sizeof(cl_uint), (void *)&srcLogicalBlockSize);
        openCLExecuteKernel(cxt, &keypoint_sort_prog, kernelname[1], globalThreads, localThreads, args, -1, -1, build_opt_buf);
    }
    //  If there are an odd number of merges, then the output data is sitting in the temp buffer.  We need to copy
    //  the results back into the input array
    if( numMerges & 1 )
    {
        tmpKeypointBuffer.copyTo(keypoints);
    }
    std::cout << "TIME: " << (clock() - time) / (double)CLOCKS_PER_SEC << std::endl;
}

static void removeDuplicated_OCL(const oclMat& keypointsIn, oclMat& keypointsOut, const oclMat& duplicateOrder)
{
    CV_Assert(keypointsIn.cols > 0);

    int cn = keypointsIn.channels();
    int depth = keypointsIn.depth();

    size_t localThreads[3] = {256, 1, 1};
    size_t globalThreads[3] = {divUp(keypointsIn.cols, localThreads[0]) * localThreads[0],
                               1,
                               1};

    const char * const channelMap[] = { "", "", "2", "4", "4" };
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    String buildOptions = format("-D T=%s%s", typeMap[depth], channelMap[cn]);

    Context *clCxt = Context::getContext();
    String kernelName = "remove_duplicated";
    std::vector< std::pair<size_t, const void *> > args;

    ensureSizeIsEnough(keypointsIn.size(), keypointsIn.type(), keypointsOut);

    int keypointsInStep = keypointsIn.step / keypointsIn.elemSize();
    int keypointsOutStep = keypointsOut.step / keypointsOut.elemSize();

    int counter = 0;
    int err = CL_SUCCESS;
    cl_mem counterCL = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(),
                                    CL_MEM_COPY_HOST_PTR, sizeof(int),
                                    &counter, &err);

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypointsIn.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsInStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypointsOut.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsOutStep));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&duplicateOrder.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&duplicateOrder.cols));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&counterCL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsIn.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&keypointsIn.rows));

    openCLExecuteKernel(clCxt, &keypoint_sort_prog, kernelName, globalThreads, localThreads, args,
                        -1, -1, buildOptions.c_str());

    clEnqueueReadBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(),
                        counterCL, CL_TRUE, 0, sizeof(int), &counter, 0, NULL, NULL);
    clReleaseMemObject(counterCL);

    keypointsOut.cols = counter;
}

static void removeDuplicated(oclMat& keypointsIn, oclMat& keypointsOut, const oclMat& keyOrder, const oclMat& duplicateOrder)
{
    CV_Assert(keypointsIn.cols > 0);

    sortKeypoints_OCL(keypointsIn, keyOrder, false);
    removeDuplicated_OCL(keypointsIn, keypointsOut, duplicateOrder);
    std::cout << "COLS3: " << keypointsOut.cols << std::endl;
}

static void retainBest(oclMat& keypoints, const int nFeatures, const oclMat& keyOrder)
{
    CV_Assert(keypoints.cols > 0);

    sortKeypoints_OCL(keypoints, keyOrder, true);
    keypoints.cols = nFeatures;
}



static void calcDescriptors(const std::vector<oclMat>& gpyr, const oclMat& keypoints, std::vector<int>& keypointsPerOctave,
                            oclMat& descriptors, const int nOctaves, int nOctaveLayers, int firstOctave)
{
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

    std::cout << "Keypoints: " << keypoints.cols << std::endl;
    std::cout << "Descriptors: " << descriptors.rows << std::endl;

    for (int i = 0; i < nOctaves; i++)
        std::cout << "keypointsPerOctave: " << keypointsPerOctave[i] << std::endl;

    if (firstOctave < 0)
    {
        int negKeypoints = 0;
        std::cout << keypointsPerOctave.size() << std::endl;
        for (int i = firstOctave; i < 0; i++)
            negKeypoints = keypointsPerOctave[i-firstOctave];

        int offset = 0;

        clock_t time = clock();

        // First pass, calculates descriptors for keypoints with positive octaves.
        for (int o = 0; o < nOctaves+firstOctave; o++)
        {
            int numKp = keypointsPerOctave[o-firstOctave];
            int pyrOctave = o-firstOctave;
            float scale = 1.f/(1 << o);

            std::cout << "pyrOctave: " << pyrOctave << std::endl;
            std::cout << "numKp: " << numKp << std::endl;

            calcDescriptors_OCL(gpyr, keypoints, descriptors, pyrOctave, nOctaveLayers, scale, offset, numKp, d, n);
            offset += numKp;
        }

        // Second pass, calculates descriptors for keypoints with negative octaves.
        // As negative octaves are stored as 2's complements, the ordered keypoints vector has
        // the negative octaves in the end, thus, they are calculated later.
        for (int o = 0; o < abs(firstOctave); o++)
        {
            int numKp = keypointsPerOctave[o];
            int pyrOctave = o;
            float scale = static_cast<float>(1 << (o-firstOctave));

            std::cout << "pyrOctave: " << pyrOctave << std::endl;
            std::cout << "numKp: " << numKp << std::endl;

            calcDescriptors_OCL(gpyr, keypoints, descriptors, pyrOctave, nOctaveLayers, scale, offset, numKp, d, n);
            offset += numKp;
        }

        std::cout << "calcDescriptors: " << (clock() - time) / (double)CLOCKS_PER_SEC << std::endl;
    }
    else
    {
        std::cout << keypointsPerOctave.size() << std::endl;

        int offset = 0;

        clock_t time = clock();

        // First pass, calculates descriptors for keypoints with positive octaves.
        for (int o = 0; o < nOctaves; o++)
        {
            int numKp = keypointsPerOctave[o];
            int pyrOctave = o;
            float scale = 1.f/(1 << o);

            std::cout << "pyrOctave: " << pyrOctave << std::endl;
            std::cout << "numKp: " << numKp << std::endl;

            calcDescriptors_OCL(gpyr, keypoints, descriptors, pyrOctave, nOctaveLayers, scale, offset, numKp, d, n);
            offset += numKp;
            //offset += keypointsPerOctave[o-firstOctave];
        }
    }
}

//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
//void SIFT_OCL::findScaleSpaceExtrema(const std::vector<oclMat>& gauss_pyr, const std::vector<oclMat>& dog_pyr,
//                                     std::vector<KeyPoint>& keypoints) const
void SIFT_OCL::findScaleSpaceExtrema(const std::vector<oclMat>& gauss_pyr, const std::vector<oclMat>& dog_pyr,
                                     const oclMat& mask, oclMat& keypoints, std::vector<int>& keypointsPerOctave,
                                     const int maxKeypoints, const int firstOctave) const
{
    int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers + 3);
    int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
    const int n = SIFT_ORI_HIST_BINS;
    float hist[n];
    int totalKeypoints = 0;
    keypointsPerOctave.resize(nOctaves);

    clock_t extremaTime = 0, adjustTime = 0, oriTime = 0;

    int extrema = 0, adjusted = 0;

    for( int o = 0; o < nOctaves; o++ )
    {
        oclMat extremaKeypoints, adjustedKeypoints;

        // Only X and Y coordinates and layer information is extracted on find extrema process
        ensureSizeIsEnough(LAYER_ROW+1,maxKeypoints,CV_32FC1,extremaKeypoints);
        extremaKeypoints.setTo(Scalar::all(0));

        ensureSizeIsEnough(ROWS_COUNT,maxKeypoints,CV_32FC1,adjustedKeypoints);
        adjustedKeypoints.setTo(Scalar::all(0));
        int octaveKeypoints = 0;

        std::cout << "maxKeypoints: " << maxKeypoints << std::endl;

        clock_t time;

        for( int i = 1; i <= nOctaveLayers; i++ )
        {
            int idx = o*(nOctaveLayers+2)+i;
            const oclMat& img = dog_pyr[idx];
            const oclMat& prev = dog_pyr[idx-1];
            const oclMat& next = dog_pyr[idx+1];

            time = clock();
            findExtrema_OCL(prev, img, next, extremaKeypoints, octaveKeypoints, o, i, maxKeypoints, threshold,
                            nOctaveLayers, static_cast<float>(contrastThreshold),
                            static_cast<float>(edgeThreshold), static_cast<float>(sigma));
            extremaTime += clock() - time;
        }

        std::cout << "extremaOctave: " << octaveKeypoints << std::endl;
        extrema += octaveKeypoints;

        time = clock();
        int adjustedOctaveKeypoints = 0;
        if (octaveKeypoints > 0)
            adjustLocalExtrema_OCL(dog_pyr, extremaKeypoints, adjustedKeypoints, octaveKeypoints, maxKeypoints,
                                   adjustedOctaveKeypoints, o, nOctaveLayers, static_cast<float>(contrastThreshold),
                                   static_cast<float>(edgeThreshold), static_cast<float>(sigma));
        adjustTime += clock() - time;
        std::cout << "adjustedOctave: " << adjustedOctaveKeypoints << std::endl;
        adjusted += adjustedOctaveKeypoints;

        octaveKeypoints = totalKeypoints;
        if (adjustedOctaveKeypoints > 0)
            calcOrientationHist_OCL(gauss_pyr, adjustedKeypoints, keypoints, adjustedOctaveKeypoints, totalKeypoints, o, nOctaveLayers, firstOctave);
        oriTime += clock() - time;

        octaveKeypoints = totalKeypoints - octaveKeypoints;
        std::cout << "octaveKeypoints: " << octaveKeypoints << std::endl;

        keypointsPerOctave[o] = octaveKeypoints;
    }

    std::cout << "Extrema: " << extrema << std::endl;
    std::cout << "Adjusted: " << adjusted << std::endl;
    std::cout << "extremaTime: " << extremaTime / (double)CLOCKS_PER_SEC << std::endl;
    std::cout << "adjustTime: " << adjustTime / (double)CLOCKS_PER_SEC << std::endl;
    std::cout << "oriTime: " << oriTime / (double)CLOCKS_PER_SEC << std::endl;

    std::cout << totalKeypoints << std::endl;
    //keypoints.cols = totalKeypoints;
}


cv::ocl::SIFT_OCL::SIFT_OCL(int _nfeatures, int _nOctaveLayers,
                            double _contrastThreshold, double _edgeThreshold, double _sigma,
                            double _keypointsRatio) :
                            nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
                            contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold),
                            sigma(_sigma), keypointsRatio(_keypointsRatio)
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


void cv::ocl::SIFT_OCL::operator()(const oclMat& image, const oclMat& mask,
                                   oclMat& keypoints, oclMat& descriptors)
{
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    int maxKeypoints = std::min(static_cast<int>(image.size().area() * keypointsRatio), 65535);
    std::cout << maxKeypoints << std::endl;

    ensureSizeIsEnough(ROWS_COUNT, maxKeypoints, CV_32FC1, keypoints);
    keypoints.setTo(Scalar::all(0));

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( !mask.empty() && mask.type() != CV_8UC1 )
        CV_Error( Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

    if( nOctaveLayers > 3 )
        CV_Error( Error::StsBadArg, "current implementation supports a maximum of 3 octave layers" );

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

    //double t, tf = getTickFrequency();
    //t = (double)getTickCount();
    clock_t pyrTime = clock();
    buildGaussianPyramid(base, gpyr, nOctaves);
    buildDoGPyramid(gpyr, dogpyr);
    std::cout << "pyrTime: " << pyrTime / (double)CLOCKS_PER_SEC << std::endl;

    std::vector<int> keypointsPerOctave;
    findScaleSpaceExtrema(gpyr, dogpyr, mask, keypoints, keypointsPerOctave, maxKeypoints, firstOctave);

    // Define keypoint
    const int key_elements = 6;
    int keys[key_elements] = {X_ROW, Y_ROW, SIZE_ROW, ANGLE_ROW, RESPONSE_ROW, OCTAVE_ROW};
    oclMat keyOrder(1,key_elements,CV_32SC1,keys);
    const int duplicate_elements = 4;
    int duplicate[duplicate_elements] = {X_ROW, Y_ROW, ANGLE_ROW, SIZE_ROW};
    oclMat duplicateOrder(1,duplicate_elements,CV_32SC1,duplicate);

    oclMat tmpKeypoints;
    removeDuplicated(keypoints, tmpKeypoints, keyOrder, duplicateOrder);
    std::cout << "Non-duplicated: " << tmpKeypoints.cols << std::endl;

    if (nfeatures > 0)
    {
        const int resp_elements = 1;
        int resp_keys[resp_elements] = {RESPONSE_ROW};
        oclMat respOrder(1,resp_elements,CV_32SC1,resp_keys);
        retainBest(tmpKeypoints, nfeatures, respOrder);
    }

    // Copy non-duplicated (and possibly only best retained features) to final keypoints matrix
    tmpKeypoints.copyTo(keypoints);
    tmpKeypoints.release();

    // if( _descriptors.needed() )

//    const int octv_elements = 1;
//    int octv_keys[octv_elements] = {OCTAVE_ROW};
//    oclMat octvOrder(1,octv_elements,CV_32SC1,octv_keys);
//    sortKeypoints_OCL(keypoints, octvOrder, false);

    // Reorder keypoints by octaves, allowing the descriptor calculator kernel to be run
    // individually for each octave
    const int octv_elements = 3;
    int octv_keys[octv_elements] = {OCTAVE_ROW, X_ROW, Y_ROW};
    oclMat octvOrder(1,octv_elements,CV_32SC1,octv_keys);
    sortKeypoints_OCL(keypoints, octvOrder, false);

    std::cout << "Keypoints: " << keypoints.cols << std::endl;

    int dsize = descriptorSize();
    ensureSizeIsEnough(keypoints.cols, dsize, CV_32FC1, descriptors);

    calcDescriptors(gpyr, keypoints, keypointsPerOctave, descriptors, nOctaves, nOctaveLayers, firstOctave);

//    Mat desc;
//    desc = descriptors;
//    Mat kp = keypoints;
//
//    for (int j = 8; j < 12; j++)
//    {
//        std::cout << kp.at<float>(X_ROW,j) << " " << kp.at<float>(Y_ROW,j) << " " << kp.at<float>(ANGLE_ROW,j) << std::endl;
//        std::cout << "[";
//        for (int i = 0; i < desc.cols; i++)
//            std::cout << desc.at<float>(j,i) << ", ";
//        std::cout << "]" << std::endl;
//    }
////
//    std::cout << kp.at<float>(X_ROW,288) << " " << kp.at<float>(Y_ROW,288) << " " << kp.at<float>(ANGLE_ROW,288) << std::endl;
//    std::cout << "[";
//    for (int i = 0; i < desc.cols; i++)
//        std::cout << desc.at<float>(288,i) << ", ";
//    std::cout << "]" << std::endl;

//    Mat tmp;
//    tmp = keypoints;
//    for (int c = 1; c < tmp.cols; c++)
//    {
//        for (int r = 0; r < ROWS_COUNT; r++)
//        {
//            std::cout << tmp.at<float>(r,c) << " ";
//        }
//        std::cout << std::endl;
//    }

//    if( !useProvidedKeypoints )
//    {
//        //t = (double)getTickCount();
//        findScaleSpaceExtrema(gpyr, dogpyr, keypoints, maxKeypoints);
//
//        const int key_elements = 6;
//        int keys[key_elements] = {X_ROW, Y_ROW, SIZE_ROW, ANGLE_ROW, RESPONSE_ROW, OCTAVE_ROW};
//        oclMat keyOrder(1,key_elements,CV_32SC1,key_order);
//        sortKeypoints(keypoints, keyOrder, ROWS_COUNT, false);
////        KeyPointsFilter::removeDuplicated( keypoints );
////
////        if( nfeatures > 0 )
////            KeyPointsFilter::retainBest(keypoints, nfeatures);
////        //t = (double)getTickCount() - t;
////        //printf("keypoint detection time: %g\n", t*1000./tf);
////
////        if( firstOctave < 0 )
////            for( size_t i = 0; i < keypoints.size(); i++ )
////            {
////                KeyPoint& kpt = keypoints[i];
////                float scale = 1.f/(float)(1 << -firstOctave);
////                kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
////                kpt.pt *= scale;
////                kpt.size *= scale;
////            }
////
////        if( !mask.empty() )
////            KeyPointsFilter::runByPixelsMask( keypoints, mask );
//    }
////    else
////    {
////        // filter keypoints by mask
////        //KeyPointsFilter::runByPixelsMask( keypoints, mask );
////    }
////
////    if( _descriptors.needed() )
////    {
////        //t = (double)getTickCount();
////        int dsize = descriptorSize();
////        _descriptors.create((int)keypoints.size(), dsize, CV_32F);
////        Mat descriptors = _descriptors.getMat();
////
////        calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
////        //t = (double)getTickCount() - t;
////        //printf("descriptor extraction time: %g\n", t*1000./tf);
////    }
}

void cv::ocl::SIFT_OCL::operator()(const oclMat& img, const oclMat& mask, std::vector<KeyPoint>& keypoints, oclMat& descriptors)
{
    (*this)(img, mask, d_keypoints_, descriptors);
    downloadKeyPoints(d_keypoints_, keypoints);
}

void cv::ocl::SIFT_OCL::downloadKeyPoints(const oclMat &d_keypoints, std::vector<KeyPoint>& keypoints)
{
    if (d_keypoints.empty())
    {
        keypoints.clear();
        return;
    }

    Mat h_keypoints(d_keypoints);

    convertKeyPoints(h_keypoints, keypoints);
}

void cv::ocl::SIFT_OCL::convertKeyPoints(const Mat &d_keypoints, std::vector<KeyPoint>& keypoints)
{
    if (d_keypoints.empty())
    {
        keypoints.clear();
        return;
    }

    CV_Assert(d_keypoints.type() == CV_32FC1 && d_keypoints.rows == ROWS_COUNT);

    const float* x_ptr = d_keypoints.ptr<float>(X_ROW);
    const float* y_ptr = d_keypoints.ptr<float>(Y_ROW);
    const float* layer_ptr = d_keypoints.ptr<float>(LAYER_ROW);
    const float* response_ptr = d_keypoints.ptr<float>(RESPONSE_ROW);
    const float* angle_ptr = d_keypoints.ptr<float>(ANGLE_ROW);
    const float* octave_ptr = d_keypoints.ptr<float>(OCTAVE_ROW);
    const float* size_ptr = d_keypoints.ptr<float>(SIZE_ROW);

    keypoints.resize(d_keypoints.cols);

    for (int i = 0; i < d_keypoints.cols; ++i)
    {
        KeyPoint kp;

        kp.pt.x = x_ptr[i];
        kp.pt.y = y_ptr[i];
        kp.response = response_ptr[i];
        kp.angle = angle_ptr[i];
        kp.size = size_ptr[i];

        // Pack keypoint octave and store
        int octave = static_cast<int>(octave_ptr[i]);
        octave |= (static_cast<int>(layer_ptr[i]) << 8);
        kp.octave = octave;

        keypoints[i] = kp;
    }
}

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