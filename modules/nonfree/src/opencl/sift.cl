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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#define CV_PI M_PI
#else
#define CV_PI M_PI_F
#endif

#define X_ROW 0
#define Y_ROW 1
#define RESPONSE_ROW 2
#define ANGLE_ROW 3
#define OCTAVE_ROW 4
#define SIZE_ROW 5
#define ROWS_COUNT 6

#define SIFT_IMG_BORDER = 5;

////////////////////////////////////////////////////////////////////////////////////////////////////////
// HarrisResponses

__kernel
void findExtrema(__global const float* prev,
                 __global const float* center,
                 __global const float* next,
                 __global float* keypoints,
                 const int img_rows,
                 const int img_cols,
                 const int prev_step,
                 const int center_step,
                 const int next_step,
                 const int keypoints_step,
                 const int max_keypoints)
{
    __local int smem0[8 * 32];
    __local int smem1[8 * 32];
    __local int smem2[8 * 32];

    const int ptidx = mad24(get_group_id(0), get_local_size(1), get_local_id(1));

    if (ptidx < npoints)
    {
        const int pt_x = keypoints[mad24(keypoints_step, X_ROW, ptidx)];
        const int pt_y = keypoints[mad24(keypoints_step, Y_ROW, ptidx)];

        const int r = blockSize / 2;
        const int x0 = pt_x - r;
        const int y0 = pt_y - r;

        int a = 0, b = 0, c = 0;

        for (int ind = get_local_id(0); ind < blockSize * blockSize; ind += get_local_size(0))
        {
            const int i = ind / blockSize;
            const int j = ind % blockSize;

            int center = mad24(y0+i, img_step, x0+j);

            int Ix = (img[center+1] - img[center-1]) * 2 +
                     (img[center-img_step+1] - img[center-img_step-1]) +
                     (img[center+img_step+1] - img[center+img_step-1]);

            int Iy = (img[center+img_step] - img[center-img_step]) * 2 +
                     (img[center+img_step-1] - img[center-img_step-1]) +
                     (img[center+img_step+1] - img[center-img_step+1]);

            a += Ix * Ix;
            b += Iy * Iy;
            c += Ix * Iy;
        }

        __local int* srow0 = smem0 + get_local_id(1) * get_local_size(0);
        __local int* srow1 = smem1 + get_local_id(1) * get_local_size(0);
        __local int* srow2 = smem2 + get_local_id(1) * get_local_size(0);

        reduce_32(srow0, &a, get_local_id(0));
        reduce_32(srow1, &b, get_local_id(0));
        reduce_32(srow2, &c, get_local_id(0));

        if (get_local_id(0) == 0)
        {
            float scale = (1 << 2) * blockSize * 255.0f;
            scale = 1.0f / scale;
            const float scale_sq_sq = scale * scale * scale * scale;

            float response = ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq;
            keypoints[mad24(keypoints_step, RESPONSE_ROW, ptidx)] = response;
        }
    }
}
