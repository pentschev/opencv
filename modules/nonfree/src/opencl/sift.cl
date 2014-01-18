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

// TODO
//
// 1) Vectorize implementation to reduce some clock cycles.


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

// Define keypoint parameter rows
#define X_ROW 0
#define Y_ROW 1
#define LAYER_ROW 2
#define RESPONSE_ROW 3
#define ANGLE_ROW 4
#define OCTAVE_ROW 5
#define SIZE_ROW 6
#define ROWS_COUNT 7

#define SIFT_IMG_BORDER 5
#define SIFT_MAX_INTERP_STEPS 5

#define MAX_ORDERING_ELEMENTS 10

// TODO: Check how to do this properly
#define SIFT_FIXPT_SCALE 1

// default number of bins in histogram for orientation assignment
#define SIFT_ORI_HIST_BINS 36

// determines gaussian sigma for orientation assignment
#define SIFT_ORI_SIG_FCTR 1.5f

// determines the radius of the region used in orientation assignment
#define SIFT_ORI_RADIUS 3 * SIFT_ORI_SIG_FCTR

// orientation magnitude relative to max that results in new feature
#define SIFT_ORI_PEAK_RATIO 0.8f

// determines the size of a single descriptor orientation histogram
#define SIFT_DESCR_SCL_FCTR 3.f

// threshold on magnitude of elements of descriptor vector
#define SIFT_DESCR_MAG_THR 0.2f

// factor used to convert floating-point descriptor to unsigned char
#define SIFT_INT_DESCR_FCTR 512.f

#ifdef CPU
void reduce_36_fmax(volatile __local float * data, int tid)
{
#define op(A, B) fmax(*A,B)

    for(int i = 32; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            data[tid] = fmax(data[tid], smem[tid + i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#undef op
}
#else
void reduce_36_fmax_local(__local float * data, int tid)
{
#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif

    if (tid < 16)
    {
        data[tid] = fmax(data[tid], data[tid + 16]);
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        data[tid] = fmax(data[tid], data[tid + 8]);
#if WAVE_SIZE < 8
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
#endif
        data[tid] = fmax(data[tid], data[tid + 4]);
#if WAVE_SIZE < 4
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)
    {
#endif
        data[tid] = fmax(data[tid], data[tid + 2]);
        data[tid+32] = fmax(data[tid+32], data[tid+32 + 2]);
#if WAVE_SIZE < 2
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        data[tid] = fmax(data[tid], data[tid + 1]);
        data[tid+32] = fmax(data[tid+32], data[tid+32 + 1]);
        data[tid] = fmax(data[tid], data[tid + 32]);
    }
#undef WAVE_SIZE
}
#endif


void atomic_add_local_float(volatile __local float *src, const float val) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;

    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *src;
        newVal.floatVal = prevVal.floatVal + val;
    } while (atomic_cmpxchg((volatile __local unsigned int *)src, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

// Gaussian elimination with pivoting to solve a 3-dimensional linear equation system
static void gaussian_elimination_with_pivoting(float* A,
                                               float* b,
                                               float* X,
                                               const int n)
{
#ifdef DOUBLE_SUPPORT
    double r, tmp;
#else
    float r, tmp;
#endif

#define AIDX(Y,X) mad24(Y,n,X)

    for (int i = 0; i < n; i++)
        X[i] = 0;

    for (int i = 0; i < n-1; i++)
    {
        for (int j = i+1; j < n; j++)
        {
            r = A[AIDX(j,i)] / A[AIDX(i,i)];

            for (int k = i; k < n; k++)
                A[AIDX(j,k)] -= r * A[AIDX(i,k)];

            b[j] -= r * b[i];
        }
    }

    X[n-1] = b[n-1] / A[AIDX(n-1,n-1)];
    for (int i = n-2; i >= 0; i--)
    {
        tmp = b[i];
        for (int j = i+1; j < n; j++) {
            tmp -= (A[AIDX(i,j)] * X[j]);
        }
        X[i] = tmp / A[AIDX(i,i)];
    }

#undef AIDX
}

/////////////////////////////////////////// KERNELS ////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////
// calcSIFTDescriptor

__kernel
void calcSIFTDescriptor(__global const T* layer1, const int layer1_step,
                        __global const T* layer2, const int layer2_step,
                        __global const T* layer3, const int layer3_step,
                        const int rows, const int cols,
                        __global const float* keypoints, const int keypoints_step,
                        __global float* descriptors, const int descriptors_step,
                        const int n_octave_layers, const int keypoints_offset,
                        const int total_keypoints, const int d,
                        const int n, const float scale)
{
    const int hidx =  mad24(get_group_id(0), get_local_size(0), get_local_id(0));
    const int kpidx = hidx + keypoints_offset;
    const int lid_y = get_local_id(1);
    const int lsz_y = get_local_size(1);

    __local float hist[360];
    __local float desc[128];

    if (kpidx >= keypoints_offset+total_keypoints)
        return;

#define KP(Y,X) keypoints[mad24(Y,keypoints_step,X)]
#define IMG(Y,X) img[mad24(Y,img_step,X)]
#define DESC(Y,X) descriptors[mad24(Y,descriptors_step,X)]

    const int layer = round(KP(LAYER_ROW, kpidx));
    float angle = 360.f - KP(ANGLE_ROW, kpidx);
    if (fabs(angle - 360.f) < FLT_EPSILON)
        angle = 0.f;
    const float ori = angle;
    const float size = KP(SIZE_ROW, kpidx);
    const int pt_x = round(KP(X_ROW, kpidx) * scale);
    const int pt_y = round(KP(Y_ROW, kpidx) * scale);

    // Points img to correct Gaussian pyramid layer
    __global const T* img;
    int img_step;

    if (layer == 1)
    {
        img = layer1;
        img_step = layer1_step;
    }
    else if (layer == 2)
    {
        img = layer2;
        img_step = layer2_step;
    }
    else if (layer == 3)
    {
        img = layer3;
        img_step = layer3_step;
    }

    float cos_t = cospi(ori/180.f);
    float sin_t = sinpi(ori/180.f);
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * size * scale * 0.5f;
    int radius = round(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);

    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = min(radius, (int) sqrt((float) cols*cols + rows*rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int len = (radius*2+1), histlen = (d+2)*(d+2)*(n+2);

    for (int i = lid_y; i < histlen; i += lsz_y)
        hist[i] = 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate orientation histogram
    for (int l = lid_y; l < len*len; l += lsz_y)
    {
        int i = l / len - radius;
        int j = l % len - radius;

        float x_rot = j * cos_t - i * sin_t;
        float y_rot = j * sin_t + i * cos_t;
        float ybin = y_rot + d/2 - 0.5f;
        float xbin = x_rot + d/2 - 0.5f;

        int y = pt_y + i;
        int x = pt_x + j;

        if( ybin > -1 && ybin < d && xbin > -1 && xbin < d &&
            y > 0 && y < rows - 1 && x > 0 && x < cols - 1 )
        {
            float dx = (float)(IMG(y, x+1) - IMG(y, x-1));
            float dy = (float)(IMG(y-1, x) - IMG(y+1, x));

            float Ori = atan2(dy,dx) * 180.f / CV_PI;
            if (Ori < 0.f)
                Ori += 360.f;
            float Mag = sqrt(dx*dx+dy*dy);
            float W = exp((x_rot*x_rot + y_rot*y_rot)*exp_scale);

            float obin = (Ori - ori)*bins_per_rad;
            float mag = Mag*W;

            int r0 = floor( ybin );
            int c0 = floor( xbin );
            int o0 = floor( obin );
            ybin -= r0;
            xbin -= c0;
            obin -= o0;

            if( o0 < 0 )
                o0 += n;
            if( o0 >= n )
                o0 -= n;

            // histogram update using tri-linear interpolation
            float v_r1 = mag*ybin, v_r0 = mag - v_r1;
            float v_rc11 = v_r1*xbin, v_rc10 = v_r1 - v_rc11;
            float v_rc01 = v_r0*xbin, v_rc00 = v_r0 - v_rc01;
            float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
            float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
            float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
            float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

            int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
            atomic_add_local_float(&hist[idx], v_rco000);
            atomic_add_local_float(&hist[idx+1], v_rco001);
            atomic_add_local_float(&hist[idx+(n+2)], v_rco010);
            atomic_add_local_float(&hist[idx+(n+3)], v_rco011);
            atomic_add_local_float(&hist[idx+(d+2)*(n+2)], v_rco100);
            atomic_add_local_float(&hist[idx+(d+2)*(n+2)+1], v_rco101);
            atomic_add_local_float(&hist[idx+(d+3)*(n+2)], v_rco110);
            atomic_add_local_float(&hist[idx+(d+3)*(n+2)+1], v_rco111);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // finalize histogram, since the orientation histograms are circular
    for (int l = lid_y; l < d*d; l += lsz_y)
    {
        int i = l / d;
        int j = l % d;

        int idx = ((i+1)*(d+2) + (j+1))*(n+2);
        atomic_add_local_float(&hist[idx], hist[idx+n]);
        atomic_add_local_float(&hist[idx+1], hist[idx+n+1]);

        for (int k = 0; k < n; k++)
            desc[(i*d + j)*n + k] = hist[idx+k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid_y == 0)
    {
        float nrm2 = 0;
        len = d*d*n;
        for( int k = 0; k < len; k++ )
            nrm2 += desc[k]*desc[k];

        float thr = sqrt(nrm2)*SIFT_DESCR_MAG_THR;
        nrm2 = 0;
        for( int i = 0; i < len; i++ )
        {
            float val = min(desc[i], thr);
            desc[i] = val;
            nrm2 += val*val;
        }
        nrm2 = SIFT_INT_DESCR_FCTR/max(sqrt(nrm2), FLT_EPSILON);

        for( int k = 0; k < len; k++ )
            DESC(kpidx, k) = round(desc[k]*nrm2);
    }

#undef KP
#undef IMG
#undef DESC
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// calcOrientationHist

// Computes a gradient orientation histogram at a specified pixel
__kernel
void calcOrientationHist(__global const T* layer1, const int layer1_step,
                         __global const T* layer2, const int layer2_step,
                         __global const T* layer3, const int layer3_step,
                         const int img_rows, const int img_cols,
                         __global float* keypoints_in, const int keypoints_in_step,
                         __global float* keypoints_out, const int keypoints_out_step,
                         __global int* counter, const int num_keypoints,
                         const int first_octave)
{
    const int kpidx = mad24(get_group_id(0), get_local_size(0), get_local_id(0));
    const int inst = mad24(get_group_id(1), get_local_size(1), get_local_id(1));
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lsz_x = get_local_size(0);
    const int lsz_y = get_local_size(1);

    const int n = SIFT_ORI_HIST_BINS;

    const int hist_step = SIFT_ORI_HIST_BINS;
    const int temphist_step = SIFT_ORI_HIST_BINS+4;
    __local float hist[SIFT_ORI_HIST_BINS*4];
    __local float temphist[(SIFT_ORI_HIST_BINS+4)*4];

#define KPIN(A,B) keypoints_in[mad24(A,keypoints_in_step,B)]
#define KPOUT(A,B) keypoints_out[mad24(A,keypoints_out_step,B)]
#define HIST(A,B) hist[mad24(A,hist_step,B)]
#define TEMPHIST(A,B) temphist[mad24(A,temphist_step,B)]
#define IMG(A,B) img[mad24(A,img_step,B)]

    if (kpidx < num_keypoints)
    {
        // Load keypoint information
        const float resp = KPIN(RESPONSE_ROW,kpidx);
        const int octave = KPIN(OCTAVE_ROW,kpidx);
        const int layer = KPIN(LAYER_ROW,kpidx);
        const float real_x = KPIN(X_ROW,kpidx);
        const float real_y = KPIN(Y_ROW,kpidx);
        const int pt_x = (int)round(KPIN(X_ROW,kpidx) / (1 << octave));
        const int pt_y = (int)round(KPIN(Y_ROW,kpidx) / (1 << octave));
        const float size = KPIN(SIZE_ROW,kpidx);

        // Calculate auxiliar parameters
        const float scl_octv = size*0.5f / (1 << octave);
        const int radius = (int)round(SIFT_ORI_RADIUS * scl_octv);
        const float sigma = SIFT_ORI_SIG_FCTR * scl_octv;
        const int len = (radius*2+1);
        const float expf_scale = -1.f/(2.f * sigma * sigma);

        // Points img to correct Gaussian pyramid layer
        __global const T* img;
        int img_step;

        if (layer == 1)
        {
            img = layer1;
            img_step = layer1_step;
        }
        else if (layer == 2)
        {
            img = layer2;
            img_step = layer2_step;
        }
        else if (layer == 3)
        {
            img = layer3;
            img_step = layer3_step;
        }

        // Initialize temporary histogram
        for (int i = lid_y; i < SIFT_ORI_HIST_BINS+4; i += lsz_y)
            TEMPHIST(lid_x, i) = 0.f;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Calculate orientation histogram
        for (int l = lid_y; l < len*len; l += lsz_y)
        {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = pt_y + i;
            int x = pt_x + j;
            if (y <= 0 || y >= img_rows - 1 ||
                x <= 0 || x >= img_cols - 1)
                continue;

            float dx = (float)(IMG(y,x+1) - IMG(y,x-1));
            float dy = (float)(IMG(y-1,x) - IMG(y+1,x));

            float Ori = atan2(dy,dx) * 180.f / CV_PI;
            float Mag = sqrt(dx*dx+dy*dy);
            float W = exp((i*i + j*j)*expf_scale);

            int bin = round((n/360.f)*Ori);
            if (bin >= n)
                bin -= n;
            if (bin < 0)
                bin += n;

            atomic_add_local_float(&TEMPHIST(lid_x, bin+2), W*Mag);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Define histogram borders
        if (lid_y == 0)
        {
            TEMPHIST(lid_x, 0) = TEMPHIST(lid_x, n);
            TEMPHIST(lid_x, 1) = TEMPHIST(lid_x, n+1);
            TEMPHIST(lid_x, n+2) = TEMPHIST(lid_x, 2);
            TEMPHIST(lid_x, n+3) = TEMPHIST(lid_x, 3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Smooth the histogram
        for (int i = 2+lid_y; i < n+2; i += lsz_y)
        {
            HIST(lid_x, i-2) = (TEMPHIST(lid_x, i-2) + TEMPHIST(lid_x, i+2))*(1.f/16.f) +
                                                   (TEMPHIST(lid_x, i-1) + TEMPHIST(lid_x, i+1))*(4.f/16.f) +
                                                   TEMPHIST(lid_x, i)*(6.f/16.f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Find maximum histogram value by reducing the 36-element histogram
        reduce_36_fmax_local(&HIST(lid_x, 0), lid_y);
        barrier(CLK_LOCAL_MEM_FENCE);
        float maxval = HIST(lid_x, 0);
        float mag_thr = (float)(maxval * SIFT_ORI_PEAK_RATIO);

        // As reduction is a destructive operation, we have to copy the data back
        for (int i = 2+lid_y; i < n+2; i += lsz_y)
        {
            HIST(lid_x, i-2) = (TEMPHIST(lid_x, i-2) + TEMPHIST(lid_x, i+2))*(1.f/16.f) +
                                                   (TEMPHIST(lid_x, i-1) + TEMPHIST(lid_x, i+1))*(4.f/16.f) +
                                                   TEMPHIST(lid_x, i)*(6.f/16.f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Tests for multiple orientations and store keypoint
        for (int j = lid_y; j < n; j += lsz_y)
        {
            int l = j > 0 ? j - 1 : n - 1;
            int r2 = j < n-1 ? j + 1 : 0;

            if (HIST(lid_x, j) > HIST(lid_x, l) &&
                HIST(lid_x, j) > HIST(lid_x, r2) &&
                HIST(lid_x, j) >= mag_thr)
            {
                int finalIdx = atomic_inc(counter);

                float bin = j + 0.5f * (HIST(lid_x, l)-HIST(lid_x, r2)) / (HIST(lid_x, l) - 2*HIST(lid_x, j) + HIST(lid_x, r2));
                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                float angle = 360.f - (float)((360.f/n) * bin);
                if(fabs(angle - 360.f) < FLT_EPSILON)
                    angle = 0.f;

                float new_octave = octave;
                float new_real_x = real_x;
                float new_real_y = real_y;
                float new_size = size;
                if( first_octave < 0 )
                {
                    float scale = 1.f/(float)(1 << -first_octave);
                    new_octave = (float)(((int)octave & ~255) | (((int)octave + first_octave) & 255));
                    new_real_x = real_x*scale;
                    new_real_y = real_y*scale;
                    new_size = size*scale;
                }

                KPOUT(X_ROW, finalIdx) = new_real_x;
                KPOUT(Y_ROW, finalIdx) = new_real_y;
                KPOUT(LAYER_ROW, finalIdx) = layer;
                KPOUT(RESPONSE_ROW, finalIdx) = resp;
                KPOUT(OCTAVE_ROW, finalIdx) = new_octave;
                KPOUT(SIZE_ROW, finalIdx) = new_size;
                KPOUT(ANGLE_ROW, finalIdx) = angle;
            }
        }
    }

#undef KPIN
#undef KPOUT
#undef HIST
#undef TEMPHIST
#undef IMG
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// adjustLocalExtrema

//
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
__kernel
void adjustLocalExtrema(__global const T* layer0, const int layer0_step,
                        __global const T* layer1, const int layer1_step,
                        __global const T* layer2, const int layer2_step,
                        __global const T* layer3, const int layer3_step,
                        __global const T* layer4, const int layer4_step,
                        const int img_rows, const int img_cols,
                        __global const float* keypoints_in, const int keypoints_in_step,
                        __global float* keypoints_out, const int keypoints_out_step,
                        __global int* counter, const int octave_keypoints,
                        const int max_keypoints, const int octave,
                        const int n_octave_layers, const float contrast_threshold,
                        const float edge_threshold, const float sigma)
{
    const int idx = mad24(get_group_id(0), get_local_size(0), get_local_id(0));

    if (idx < octave_keypoints)
    {

#define PREV(A,B) prev[mad24(A,prev_step,B)]
#define CENTER(A,B) center[mad24(A,center_step,B)]
#define NEXT(A,B) next[mad24(A,next_step,B)]
#define KPIN(A,B) keypoints_in[mad24(A,keypoints_in_step,B)]
#define KPOUT(A,B) keypoints_out[mad24(A,keypoints_out_step,B)]

        const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
        const float deriv_scale = img_scale*0.5f;
        const float second_deriv_scale = img_scale;
        const float cross_deriv_scale = img_scale*0.25f;

        float xi=0, xr=0, xc=0, contr=0;
        int i = 0;

        int c = KPIN(X_ROW,idx);
        int r = KPIN(Y_ROW,idx);
        int layer = KPIN(LAYER_ROW,idx);

        __global const T* prev;
        __global const T* center;
        __global const T* next;
        int prev_step, center_step, next_step;

        // Iterative 3D keypoint interpolation
        for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
        {
            if (layer == 1)
            {
                prev = layer0;
                prev_step = layer0_step;
                center = layer1;
                center_step = layer1_step;
                next = layer2;
                next_step = layer2_step;
            }
            else if (layer == 2)
            {
                prev = layer1;
                prev_step = layer1_step;
                center = layer2;
                center_step = layer2_step;
                next = layer3;
                next_step = layer3_step;
            }
            else if (layer == 3)
            {
                prev = layer2;
                prev_step = layer2_step;
                center = layer3;
                center_step = layer3_step;
                next = layer4;
                next_step = layer4_step;
            }

            float dD[3] = {(CENTER(r, c+1) - CENTER(r, c-1)) * deriv_scale,
                           (CENTER(r+1, c) - CENTER(r-1, c)) * deriv_scale,
                           (NEXT(r, c) - PREV(r, c)) * deriv_scale};

            float v2 = (float)CENTER(r, c)*2.f;
            float dxx = (CENTER(r, c+1) + CENTER(r, c-1) - v2)*second_deriv_scale;
            float dyy = (CENTER(r+1, c) + CENTER(r-1, c) - v2)*second_deriv_scale;
            float dss = (NEXT(r, c) + PREV(r, c) - v2)*second_deriv_scale;
            float dxy = (CENTER(r+1, c+1) - CENTER(r+1, c-1) -
                         CENTER(r-1, c+1) + CENTER(r-1, c-1))*cross_deriv_scale;
            float dxs = (NEXT(r, c+1) - NEXT(r, c-1) -
                         PREV(r, c+1) + PREV(r, c-1))*cross_deriv_scale;
            float dys = (NEXT(r+1, c) - NEXT(r-1, c) -
                         PREV(r+1, c) + PREV(r-1, c))*cross_deriv_scale;

            float H[9] = {dxx, dxy, dxs,
                          dxy, dyy, dys,
                          dxs, dys, dss};

            float X[3];
            gaussian_elimination_with_pivoting(H, dD, X, 3);

            xi = -X[2];
            xr = -X[1];
            xc = -X[0];

            if(fabs(xi) < 0.5f && fabs(xr) < 0.5f && fabs(xc) < 0.5f)
                break;

            if(fabs(xi) > (float)(INT_MAX/3) ||
               fabs(xr) > (float)(INT_MAX/3) ||
               fabs(xc) > (float)(INT_MAX/3))
                return;

            c += (int)round(xc);
            r += (int)round(xr);
            layer += (int)round(xi);

            if(layer < 1 || layer > n_octave_layers ||
               c < SIFT_IMG_BORDER || c >= img_cols - SIFT_IMG_BORDER  ||
               r < SIFT_IMG_BORDER || r >= img_rows - SIFT_IMG_BORDER)
                return;
        }

        // ensure convergence of interpolation
        if (i >= SIFT_MAX_INTERP_STEPS)
            return;

        float4 dD = {(CENTER(r, c+1) - CENTER(r, c-1)) * deriv_scale,
                     (CENTER(r+1, c) - CENTER(r-1, c)) * deriv_scale,
                     (NEXT(r, c) - PREV(r, c)) * deriv_scale,
                     0};
        float4 X = {xc, xr, xi, 0};
        float t = dot(dD, X);

        contr = CENTER(r, c)*img_scale + t * 0.5f;
        if(fabs(contr) * n_octave_layers < contrast_threshold)
            return;

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = CENTER(r, c)*2.f;
        float dxx = (CENTER(r, c+1) + CENTER(r, c-1) - v2)*second_deriv_scale;
        float dyy = (CENTER(r+1, c) + CENTER(r-1, c) - v2)*second_deriv_scale;
        float dxy = (CENTER(r+1, c+1) - CENTER(r+1, c-1) -
                     CENTER(r-1, c+1) + CENTER(r-1, c-1)) * cross_deriv_scale;

        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if (det <= 0 || tr*tr*edge_threshold >= (edge_threshold + 1)*(edge_threshold + 1)*det)
            return;

        int res_idx = atomic_inc(counter);

        if (res_idx < max_keypoints)
        {
            KPOUT(X_ROW,res_idx) = (c + xc) * (1 << octave);
            KPOUT(Y_ROW,res_idx) = (r + xr) * (1 << octave);
            KPOUT(RESPONSE_ROW,res_idx) = fabs(contr);
            KPOUT(OCTAVE_ROW,res_idx) = octave;
            KPOUT(LAYER_ROW,res_idx) = layer;
            KPOUT(SIZE_ROW,res_idx) = sigma*pow(2.f, (layer + xi) / n_octave_layers)*(1 << octave)*2;
        }

#undef PREV
#undef CENTER
#undef NEXT
#undef KPIN
#undef KPOUT

        return;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// findExtrema

__kernel
void findExtrema(__global const T* prev, const int prev_step,
                 __global const T* center, const int center_step,
                 __global const T* next, const int next_step,
                 const int img_rows, const int img_cols,
                 __global float* keypoints, const int keypoints_step,
                 __global int* counter, const int octave,
                 const int scale, const int max_keypoints,
                 const int threshold, const int n_octave_layers,
                 const float contrast_threshold, const float edge_threshold,
                 const float sigma)
{
    // One pixel border for each side
    __local T next_mem[(8+2) * (32+2)];
    __local T center_mem[(8+2) * (32+2)];
    __local T prev_mem[(8+2) * (32+2)];

    const int lid_j = get_local_id(0);
    const int lid_i = get_local_id(1);
    const int lsz_j = get_local_size(0);
    const int lsz_i = get_local_size(1);
    const int j = mad24(get_group_id(0), get_local_size(0), (size_t)(lid_j+SIFT_IMG_BORDER));
    const int i = mad24(get_group_id(1), get_local_size(1), (size_t)(lid_i+SIFT_IMG_BORDER));

    const int x = lid_j+1;
    const int y = lid_i+1;

    const int mem_step = lsz_j + 2;

#define MPREV(A,B) prev_mem[mad24(A,mem_step,B)]
#define MCENTER(A,B) center_mem[mad24(A,mem_step,B)]
#define MNEXT(A,B) next_mem[mad24(A,mem_step,B)]
#define PREV(A,B) prev[mad24(A,prev_step,B)]
#define CENTER(A,B) center[mad24(A,center_step,B)]
#define NEXT(A,B) next[mad24(A,next_step,B)]
#define KP(A,B) keypoints[mad24(A,keypoints_step,B)]

    if (i < img_rows-SIFT_IMG_BORDER && j < img_cols-SIFT_IMG_BORDER)
    {
        // Start by copying the parts of interest of each image to local memory
        // Copy central part
        MPREV(y, x) = PREV(i, j);
        MCENTER(y, x) = CENTER(i, j);
        MNEXT(y, x) = NEXT(i, j);

        // Copy top border
        if (lid_i == 0)
        {
            MPREV(y-1, x) = PREV(i-1, j);
            MCENTER(y-1, x) = CENTER(i-1, j);
            MNEXT(y-1, x) = NEXT(i-1, j);

            // Copy top left pixel
            if (lid_j == 0)
            {
                MPREV(y-1, x-1) = PREV(i-1, j-1);
                MCENTER(y-1, x-1) = CENTER(i-1, j-1);
                MNEXT(y-1, x-1) = NEXT(i-1, j-1);
            }
            // Copy top right pixel
            else if (lid_j == lsz_j-1)
            {
                MPREV(y-1, x+1) = PREV(i-1, j+1);
                MCENTER(y-1, x+1) = CENTER(i-1, j+1);
                MNEXT(y-1, x+1) = NEXT(i-1, j+1);
            }
        }

        // Copy left border
        if (lid_j == 0)
        {
            MPREV(y, x-1) = PREV(i, j-1);
            MCENTER(y, x-1) = CENTER(i, j-1);
            MNEXT(y, x-1) = NEXT(i, j-1);
        }

        // Copy right border
        if (lid_j == lsz_j-1 || j == img_cols-SIFT_IMG_BORDER-1)
        {
            MPREV(y, x+1) = PREV(i, j+1);
            MCENTER(y, x+1) = CENTER(i, j+1);
            MNEXT(y, x+1) = NEXT(i, j+1);
        }

        // Copy bottom border
        if (lid_i == lsz_i-1 || i == img_rows-SIFT_IMG_BORDER-1)
        {
            MPREV(y+1, x) = PREV(i+1, j);
            MCENTER(y+1, x) = CENTER(i+1, j);
            MNEXT(y+1, x) = NEXT(i+1, j);

            if (lid_j == 0)
            {
                MPREV(y+1, x-1) = PREV(i+1, j-1);
                MCENTER(y+1, x-1) = CENTER(i+1, j-1);
                MNEXT(y+1, x-1) = NEXT(i+1, j-1);
            }
            else if (lid_j == lsz_j-1 ||
                     (j == img_cols-SIFT_IMG_BORDER-1 && i == img_rows-SIFT_IMG_BORDER-1))
            {
                MPREV(y+1, x+1) = PREV(i+1, j+1);
                MCENTER(y+1, x+1) = CENTER(i+1, j+1);
                MNEXT(y+1, x+1) = NEXT(i+1, j+1);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        T val = MCENTER(y,x);

        // Perform extrema localization and store its location
        if (fabs((float)val) > threshold &&
            ((val > 0 && val >= MCENTER(y-1, x-1) && val >= MCENTER(y-1, x) &&
              val >= MCENTER(y-1, x+1) && val >= MCENTER(y, x-1) && val >= MCENTER(y, x+1) &&
              val >= MCENTER(y+1, x-1) && val >= MCENTER(y+1, x) && val >= MCENTER(y+1, x+1) &&
              val >= MPREV(y-1, x-1) && val >= MPREV(y-1, x) && val >= MPREV(y-1, x+1) &&
              val >= MPREV(y, x-1) && val >= MPREV(y, x) && val >= MPREV(y, x+1) &&
              val >= MPREV(y+1, x-1) && val >= MPREV(y+1, x) && val >= MPREV(y+1, x+1) &&
              val >= MNEXT(y-1, x-1) && val >= MNEXT(y-1, x) && val >= MNEXT(y-1, x+1) &&
              val >= MNEXT(y, x-1) && val >= MNEXT(y, x) && val >= MNEXT(y, x+1) &&
              val >= MNEXT(y+1, x-1) && val >= MNEXT(y+1, x) && val >= MNEXT(y+1, x+1)) ||
             (val < 0 && val <= MCENTER(y-1, x-1) && val <= MCENTER(y-1, x) &&
              val <= MCENTER(y-1, x+1) && val <= MCENTER(y, x-1) && val <= MCENTER(y, x+1) &&
              val <= MCENTER(y+1, x-1) && val <= MCENTER(y+1, x) && val <= MCENTER(y+1, x+1) &&
              val <= MPREV(y-1, x-1) && val <= MPREV(y-1, x) && val <= MPREV(y-1, x+1) &&
              val <= MPREV(y, x-1) && val <= MPREV(y, x) && val <= MPREV(y, x+1) &&
              val <= MPREV(y+1, x-1) && val <= MPREV(y+1, x) && val <= MPREV(y+1, x+1) &&
              val <= MNEXT(y-1, x-1) && val <= MNEXT(y-1, x) && val <= MNEXT(y-1, x+1) &&
              val <= MNEXT(y, x-1) && val <= MNEXT(y, x) && val <= MNEXT(y, x+1) &&
              val <= MNEXT(y+1, x-1) && val <= MNEXT(y+1, x) && val <= MNEXT(y+1, x+1))))
        {
            int idx = atomic_inc(counter);

            if (idx < max_keypoints)
            {
                KP(X_ROW, idx) = (float)j;
                KP(Y_ROW, idx) = (float)i;
                KP(LAYER_ROW, idx) = scale;
            }
        }
    }

#undef MPREV
#undef MCENTER
#undef MNEXT
#undef PREV
#undef CENTER
#undef NEXT
#undef KP
}
