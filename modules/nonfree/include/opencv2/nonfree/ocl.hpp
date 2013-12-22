/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
// License Agreement
// For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistribution's of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistribution's in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * The name of the copyright holders may not be used to endorse or promote products
// derived from this software without specific prior written permission.
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
//M*/

#ifndef __OPENCV_NONFREE_OCL_HPP__
#define __OPENCV_NONFREE_OCL_HPP__

#include "opencv2/ocl.hpp"

namespace cv
{
    namespace ocl
    {
        ////////////////////////////////// SIFT //////////////////////////////////
        class CV_EXPORTS SIFT_OCL
        {
        public:
            enum
            {
                X_ROW = 0,
                Y_ROW,
                RESPONSE_ROW,
                ANGLE_ROW,
                OCTAVE_ROW,
                SIZE_ROW,
                ROWS_COUNT
            };

            //! Constructor
            explicit SIFT_OCL(int nfeatures = 0, int nOctaveLayers = 3,
                              double contrastThreshold = 0.04, double edgeThreshold = 10,
                              double sigma = 1.6);

            //! returns the descriptor size in floats (128)
            int descriptorSize() const;

            //! returns the descriptor type
            int descriptorType() const;

            //! returns the default norm type
            int defaultNorm() const;

            //! finds the keypoints using SIFT algorithm
            void operator()(oclMat& img, oclMat& mask,
                            std::vector<KeyPoint>& keypoints) const;

            void buildGaussianPyramid(const oclMat& base, std::vector<oclMat>& pyr, int nOctaves) const;
            void buildDoGPyramid(const std::vector<oclMat>& pyr, std::vector<oclMat>& dogpyr) const;
//            void findScaleSpaceExtrema(const std::vector<oclMat>& gauss_pyr, const std::vector<oclMat>& dog_pyr,
//                                       std::vector<KeyPoint>& keypoints) const;
            void findScaleSpaceExtrema(const std::vector<oclMat>& gauss_pyr, const std::vector<oclMat>& dog_pyr,
                                       oclMat& keypoints) const;

//            //! finds the keypoints and computes descriptors for them using SIFT algorithm.
//            //! Optionally it can compute descriptors for the user-provided keypoints
//            void operator()(oclMat& img, oclMat& mask,
//                            std::vector<KeyPoint>& keypoints,
//                            const oclMat& descriptors = oclMat(),
//                            bool useProvidedKeypoints = false) const;

//            //! Compute the ORB features on an image
//            //! image - the image to compute the features (supports only CV_8UC1 images)
//            //! mask - the mask to apply
//            //! keypoints - the resulting keypoints
//            void operator ()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints);
//            void operator ()(const oclMat& image, const oclMat& mask, oclMat& keypoints);
//
//            //! Compute the ORB features and descriptors on an image
//            //! image - the image to compute the features (supports only CV_8UC1 images)
//            //! mask - the mask to apply
//            //! keypoints - the resulting keypoints
//            //! descriptors - descriptors array
//            void operator ()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints, oclMat& descriptors);
//            void operator ()(const oclMat& image, const oclMat& mask, oclMat& keypoints, oclMat& descriptors);
//
//            //! download keypoints from device to host memory
//            static void downloadKeyPoints(const oclMat& d_keypoints, std::vector<KeyPoint>& keypoints);
//            //! convert keypoints to KeyPoint vector
//            static void convertKeyPoints(const Mat& d_keypoints, std::vector<KeyPoint>& keypoints);
//
//            //! returns the descriptor size in bytes
//            inline int descriptorSize() const { return kBytes; }
//            inline int descriptorType() const { return CV_8U; }
//            inline int defaultNorm() const { return NORM_HAMMING; }
//
//            inline void setFastParams(int threshold, bool nonmaxSupression = true)
//            {
//                fastDetector_.threshold = threshold;
//                fastDetector_.nonmaxSupression = nonmaxSupression;
//            }
//
//            //! release temporary buffer's memory
//            void release();
//
//            //! if true, image will be blurred before descriptors calculation
//            bool blurForDescriptor;

        private:
            int nfeatures;
            int nOctaveLayers;
            double contrastThreshold;
            double edgeThreshold;
            double sigma;
            double keypointsRatio;

//            enum { kBytes = 32 };
//
//            void buildScalePyramids(const oclMat& image, const oclMat& mask);
//
//            void computeKeyPointsPyramid();
//
//            void computeDescriptors(oclMat& descriptors);
//
//            void mergeKeyPoints(oclMat& keypoints);
//
//            int nFeatures_;
//            float scaleFactor_;
//            int nLevels_;
//            int edgeThreshold_;
//            int firstLevel_;
//            int WTA_K_;
//            int scoreType_;
//            int patchSize_;
//
//            // The number of desired features per scale
//            std::vector<size_t> n_features_per_level_;
//
//            // Points to compute BRIEF descriptors from
//            oclMat pattern_;
//
//            std::vector<oclMat> imagePyr_;
//            std::vector<oclMat> maskPyr_;
//
//            oclMat buf_;
//
//            std::vector<oclMat> keyPointsPyr_;
//            std::vector<int> keyPointsCount_;
//
//            FAST_OCL fastDetector_;
//
//            Ptr<ocl::FilterEngine_GPU> blurFilter;
//
//            oclMat d_keypoints_;
//
//            oclMat uMax_;
        };

        //! Speeded up robust features, port from CUDA module.
        ////////////////////////////////// SURF //////////////////////////////////////////

        class CV_EXPORTS SURF_OCL
        {
        public:
            enum KeypointLayout
            {
                X_ROW = 0,
                Y_ROW,
                LAPLACIAN_ROW,
                OCTAVE_ROW,
                SIZE_ROW,
                ANGLE_ROW,
                HESSIAN_ROW,
                ROWS_COUNT
            };

            //! the default constructor
            SURF_OCL();
            //! the full constructor taking all the necessary parameters
            explicit SURF_OCL(double _hessianThreshold, int _nOctaves = 4,
                              int _nOctaveLayers = 2, bool _extended = false, float _keypointsRatio = 0.01f, bool _upright = false);

            //! returns the descriptor size in float's (64 or 128)
            int descriptorSize() const;
            //! returns the default norm type
            int defaultNorm() const;
            //! upload host keypoints to device memory
            void uploadKeypoints(const std::vector<cv::KeyPoint> &keypoints, oclMat &keypointsocl);
            //! download keypoints from device to host memory
            void downloadKeypoints(const oclMat &keypointsocl, std::vector<KeyPoint> &keypoints);
            //! download descriptors from device to host memory
            void downloadDescriptors(const oclMat &descriptorsocl, std::vector<float> &descriptors);
            //! finds the keypoints using fast hessian detector used in SURF
            //! supports CV_8UC1 images
            //! keypoints will have nFeature cols and 6 rows
            //! keypoints.ptr<float>(X_ROW)[i] will contain x coordinate of i'th feature
            //! keypoints.ptr<float>(Y_ROW)[i] will contain y coordinate of i'th feature
            //! keypoints.ptr<float>(LAPLACIAN_ROW)[i] will contain laplacian sign of i'th feature
            //! keypoints.ptr<float>(OCTAVE_ROW)[i] will contain octave of i'th feature
            //! keypoints.ptr<float>(SIZE_ROW)[i] will contain size of i'th feature
            //! keypoints.ptr<float>(ANGLE_ROW)[i] will contain orientation of i'th feature
            //! keypoints.ptr<float>(HESSIAN_ROW)[i] will contain response of i'th feature
            void operator()(const oclMat &img, const oclMat &mask, oclMat &keypoints);
            //! finds the keypoints and computes their descriptors.
            //! Optionally it can compute descriptors for the user-provided keypoints and recompute keypoints direction
            void operator()(const oclMat &img, const oclMat &mask, oclMat &keypoints, oclMat &descriptors,
                            bool useProvidedKeypoints = false);
            void operator()(const oclMat &img, const oclMat &mask, std::vector<KeyPoint> &keypoints);
            void operator()(const oclMat &img, const oclMat &mask, std::vector<KeyPoint> &keypoints, oclMat &descriptors,
                            bool useProvidedKeypoints = false);
            void operator()(const oclMat &img, const oclMat &mask, std::vector<KeyPoint> &keypoints, std::vector<float> &descriptors,
                            bool useProvidedKeypoints = false);

            void releaseMemory();

            // SURF parameters
            float hessianThreshold;
            int nOctaves;
            int nOctaveLayers;
            bool extended;
            bool upright;
            //! max keypoints = min(keypointsRatio * img.size().area(), 65535)
            float keypointsRatio;
            oclMat sum, mask1, maskSum, intBuffer;
            oclMat det, trace;
            oclMat maxPosBuffer;

        };
    }
}

#endif //__OPENCV_NONFREE_OCL_HPP__
