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
// 2) Global memory has been used extensively due to the high amount of data, with careful evaluation
// some parts of the algorithm might be implementable using local memory. One idea is to implement
// the kernel in such a way that each work-group will process a small number of features at a time
// (e.g., 1), but this has to be tested to evaluate if processing time results improve.


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

//#ifdef CPU
//void reduce_32(volatile __local int* smem, volatile int* val, int tid)
//{
//#define op(A, B) (*A)+(B)
//
//    smem[tid] = *val;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for(int i = 16; i > 0; i >>= 1)
//    {
//        if(tid < i)
//        {
//            smem[tid] = *val = op(val, smem[tid + i]);
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//#undef op
//}
//#else
void reduce_16_fmax(__global float * data, int tid)
{
#define op(A, B) fmax(*A,B)
//    data[tid] = *partial_reduction;
//    barrier(CLK_GLOBAL_MEM_FENCE);
#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif
    if (tid < 8)
    {
        data[tid] = fmax(data[tid], data[tid + 8]);
#if WAVE_SIZE < 8
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid < 4)
    {
#endif
        data[tid] = fmax(data[tid], data[tid + 4]);
#if WAVE_SIZE < 4
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid < 2)
    {
#endif
        data[tid] = fmax(data[tid], data[tid + 2]);
#if WAVE_SIZE < 2
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        data[tid] = fmax(data[tid], data[tid + 1]);
    }
#undef WAVE_SIZE
#undef op
}
//#endif

void reduce_36_fmax(__global float * data, int tid)
{
#define op(A, B) fmax(A,B)


//    if (tid < 4)
//    data[tid] = op(data[tid], data[tid + 32]);
//
//    barrier(CLK_GLOBAL_MEM_FENCE);

#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif
    if (tid < 8)
    {
        data[tid*2] = op(data[tid*2], data[tid*2 + 16]);
        data[tid*2+1] = op(data[tid*2+1], data[tid*2+1 + 16]);
#if WAVE_SIZE < 8
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid < 4)
    {
#endif
        data[tid*2] = op(data[tid*2], data[tid*2 + 8]);
        data[tid*2+1] = op(data[tid*2+1], data[tid*2+1 + 8]);
#if WAVE_SIZE < 4
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid < 2)
    {
#endif
        data[tid*2] = op(data[tid*2], data[tid*2 + 4]);
        data[tid*2+1] = op(data[tid*2+1], data[tid*2+1 + 4]);
//        barrier(CLK_GLOBAL_MEM_FENCE);
//        data[tid*2] = op(data[tid*2], data[tid*2 + 32]);
//        data[tid*2+1] = op(data[tid*2+1], data[tid*2+1 + 32]);
#if WAVE_SIZE < 2
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        data[tid*2] = op(data[tid*2], data[tid*2 + 2]);
        data[tid*2+1] = op(data[tid*2+1], data[tid*2+1 + 2]);
        barrier(CLK_GLOBAL_MEM_FENCE);
        data[tid] = op(data[tid], data[tid + 1]);
    }
#undef WAVE_SIZE
#undef op
}

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


void atomic_add_global_float(volatile __global float *src, const float val) {
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
    } while (atomic_cmpxchg((volatile __global unsigned int *)src, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

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

__kernel
void calcSIFTDescriptor(__global const T* layer1,
                        __global const T* layer2,
                        __global const T* layer3,
                        __global const float* keypoints,
                        __global float* descriptors,
                        __global float* hist,
                        const int n_octave_layers,
                        const int keypoints_offset,
                        const int total_keypoints,
                        const int rows,
                        const int cols,
                        const int layer1_step,
                        const int layer2_step,
                        const int layer3_step,
                        const int keypoints_step,
                        const int descriptors_step,
                        const int hist_step,
//                        float ori, float scl,
                               int d, int n)
{
    const int hidx =  mad24(get_group_id(0), get_local_size(0), get_local_id(0));
    const int kpidx = hidx + keypoints_offset;
    const int lid_y = get_local_id(1);
    const int lsz_y = get_local_size(1);

    if (kpidx >= keypoints_offset+total_keypoints)
        return;

#define KPT(Y,X) keypoints[mad24(Y,keypoints_step,X)]
#define HIST(Y,X) hist[mad24(Y,hist_step,X)]
#define IMG(Y,X) img[mad24(Y,img_step,X)]
#define DESC(Y,X) descriptors[mad24(Y,descriptors_step,X)]

    const int pt_x = round(KPT(X_ROW, kpidx));
    const int pt_y = round(KPT(Y_ROW, kpidx));
    const int layer = round(KPT(LAYER_ROW, kpidx));
    float angle = 360.f - KPT(ANGLE_ROW, kpidx);
    if (fabs(angle - 360.f) < FLT_EPSILON)
        angle = 0.f;
    const float ori = angle;
    const float size = KPT(SIZE_ROW, kpidx);
    const float scl = size*0.5f;

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

//    float cos_t = cos(ori*(float)(CV_PI/180));
//    float sin_t = sin(ori*(float)(CV_PI/180));
    float cos_t = cospi(ori/180.f);
    float sin_t = sinpi(ori/180);
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = round(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);

    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = min(radius, (int) sqrt((float) cols*cols + rows*rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int len = (radius*2+1), histlen = (d+2)*(d+2)*(n+2);

    for (int i = lid_y; i < histlen; i += lsz_y)
        HIST(hidx, i) = 0.f;
    barrier(CLK_GLOBAL_MEM_FENCE);

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
        if (y <= 0 || y >= rows - 1)
            continue;

        int x = pt_x + j;
        if (x <= 0 || x >= cols - 1)
            continue;

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
            atomic_add_global_float(&HIST(hidx, idx), v_rco000);
            atomic_add_global_float(&HIST(hidx, idx+1), v_rco001);
            atomic_add_global_float(&HIST(hidx, idx+(n+2)), v_rco010);
            atomic_add_global_float(&HIST(hidx, idx+(n+3)), v_rco011);
            atomic_add_global_float(&HIST(hidx, idx+(d+2)*(n+2)), v_rco100);
            atomic_add_global_float(&HIST(hidx, idx+(d+2)*(n+2)+1), v_rco101);
            atomic_add_global_float(&HIST(hidx, idx+(d+3)*(n+2)), v_rco110);
            atomic_add_global_float(&HIST(hidx, idx+(d+3)*(n+2)+1), v_rco111);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // finalize histogram, since the orientation histograms are circular
    for (int l = lid_y; l < d*d; l += lsz_y)
    {
        int i = l / d;
        int j = l % d;

        int idx = ((i+1)*(d+2) + (j+1))*(n+2);
        atomic_add_global_float(&HIST(hidx, idx), HIST(hidx, idx+n));
        atomic_add_global_float(&HIST(hidx, idx+1), HIST(hidx, idx+n+1));

        for (int k = 0; k < n; k++)
            DESC(kpidx, (i*d + j)*n + k) = HIST(hidx, idx+k);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    len = d*d*n;

//    __local float nrm1;
//    if(lid_y == 0)
//        nrm1 = 300000;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for (int k = lid_y; k < len; k += lsz_y)
//        atomic_add_local_float(&nrm1, DESC(kpidx, k)*DESC(kpidx, k));
//    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid_y == 0)
    {
        float nrm2 = 0;
        len = d*d*n;
        for( int k = 0; k < len; k++ )
            nrm2 += DESC(kpidx, k)*DESC(kpidx, k);

//        float nrm2 = nrm1;
        float thr = sqrt(nrm2)*SIFT_DESCR_MAG_THR;
        nrm2 = 0;
        for( int i = 0; i < len; i++ )
        {
            float val = min(DESC(kpidx, i), thr);
            DESC(kpidx, i) = val;
            nrm2 += val*val;
        }
        nrm2 = SIFT_INT_DESCR_FCTR/max(sqrt(nrm2), FLT_EPSILON);

#if 1
        for( int k = 0; k < len; k++ )
        {
            DESC(kpidx, k) = round(DESC(kpidx, k)*nrm2);
        }
#else
//        float nrm1 = 0;
//        for( k = 0; k < len; k++ )
//        {
//            dst[k] *= nrm2;
//            nrm1 += dst[k];
//        }
//        nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
//        for( k = 0; k < len; k++ )
//        {
//            dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
//        }
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// calcOrientationHist

// Computes a gradient orientation histogram at a specified pixel
__kernel
void calcOrientationHist(__global const T* layer1,
                         __global const T* layer2,
                         __global const T* layer3,
                         __global float* keypoints_in,
                         __global float* keypoints_out,
                         __global float* hist,
                         __global float* temphist,
                         __global int* counter,
                         const int num_keypoints,
                         const int first_octave,
                         const int img_rows,
                         const int img_cols,
                         const int layer1_step,
                         const int layer2_step,
                         const int layer3_step,
                         const int keypoints_in_step,
                         const int keypoints_out_step,
                         const int hist_step,
                         const int temphist_step)
{
    const int kpidx = mad24(get_group_id(0), get_local_size(0), get_local_id(0));
    const int inst = mad24(get_group_id(1), get_local_size(1), get_local_id(1));
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lsz_x = get_local_size(0);
    const int lsz_y = get_local_size(1);

    const int n = SIFT_ORI_HIST_BINS;

#define KIIDX(A,B) mad24(A,keypoints_in_step,B)
#define KOIDX(A,B) mad24(A,keypoints_out_step,B)
#define HIDX(A,B) mad24(A,hist_step,B)
#define MIDX(A,B) mad24(A,mask_step,B)
#define THIDX(A,B) mad24(A,temphist_step,B)
#define IIDX(A,B) mad24(A,img_step,B)

    if (kpidx < num_keypoints)
    {
        const float resp = keypoints_in[KIIDX(RESPONSE_ROW,kpidx)];
        const int octave = keypoints_in[KIIDX(OCTAVE_ROW,kpidx)];
        const int layer = keypoints_in[KIIDX(LAYER_ROW,kpidx)];
        const float real_x = keypoints_in[KIIDX(X_ROW,kpidx)];
        const float real_y = keypoints_in[KIIDX(Y_ROW,kpidx)];
        const int pt_x = (int)round(keypoints_in[KIIDX(X_ROW,kpidx)] / (1 << octave));
        const int pt_y = (int)round(keypoints_in[KIIDX(Y_ROW,kpidx)] / (1 << octave));
        const float size = keypoints_in[KIIDX(SIZE_ROW,kpidx)];
        const float scl_octv = size*0.5f / (1 << octave);
        const int radius = (int)round(SIFT_ORI_RADIUS * scl_octv);
        const float sigma = SIFT_ORI_SIG_FCTR * scl_octv;

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

        int len = (radius*2+1);
        float expf_scale = -1.f/(2.f * sigma * sigma);

        // Calculate orientation histogram
        for (int l = lid_y; l < len*len; l += lsz_y)
        {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = pt_y + i;
            if (y <= 0 || y >= img_rows - 1)
                continue;

            int x = pt_x + j;
            if (x <= 0 || x >= img_cols - 1)
                continue;

            float dx = (float)(img[IIDX(y,x+1)] - img[IIDX(y,x-1)]);
            float dy = (float)(img[IIDX(y-1,x)] - img[IIDX(y+1,x)]);

            float Ori = atan2(dy,dx) * 180.f / CV_PI;
            float Mag = sqrt(dx*dx+dy*dy);
            float W = exp((i*i + j*j)*expf_scale);

            int bin = round((n/360.f)*Ori);
            if (bin >= n)
                bin -= n;
            if (bin < 0)
                bin += n;

            atomic_add_global_float(temphist+THIDX(kpidx, bin+2), W*Mag);
        }

        // Define histogram borders
        if (lid_y == 0)
        {
            temphist[THIDX(kpidx, 0)] = temphist[THIDX(kpidx, n)];
            temphist[THIDX(kpidx, 1)] = temphist[THIDX(kpidx, n+1)];
            temphist[THIDX(kpidx, n+2)] = temphist[THIDX(kpidx, 2)];
            temphist[THIDX(kpidx, n+3)] = temphist[THIDX(kpidx, 3)];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Smooth the histogram
        for (int i = 2+lid_y; i < n+2; i += lsz_y)
        {
            hist[HIDX(kpidx, i-2)] = (temphist[THIDX(kpidx, i-2)] + temphist[THIDX(kpidx, i+2)])*(1.f/16.f) +
                                     (temphist[THIDX(kpidx, i-1)] + temphist[THIDX(kpidx, i+1)])*(4.f/16.f) +
                                      temphist[THIDX(kpidx, i)]*(6.f/16.f);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Reduce 36-element histogram
        // TODO: Reduce to a single operation
        reduce_16_fmax(hist+HIDX(kpidx, 0), lid_y);
        reduce_16_fmax(hist+HIDX(kpidx, 16), lid_y);
//        reduce_36_fmax(hist+HIDX(kpidx, 0), lid_y);
        barrier(CLK_GLOBAL_MEM_FENCE);
        float maxval = fmax(hist[HIDX(kpidx, 0)],hist[HIDX(kpidx, 16)]);
//        float maxval = hist[HIDX(kpidx, 0)];
        maxval = fmax(maxval, hist[HIDX(kpidx, 32)]);
        maxval = fmax(maxval, hist[HIDX(kpidx, 33)]);
        maxval = fmax(maxval, hist[HIDX(kpidx, 34)]);
        maxval = fmax(maxval, hist[HIDX(kpidx, 35)]);
        float mag_thr = (float)(maxval * SIFT_ORI_PEAK_RATIO);
        barrier(CLK_GLOBAL_MEM_FENCE);

        // As reduction is a destructive operation, we have to copy the data back
        for (int i = 2+lid_y; i < n+2; i += lsz_y)
        {
            hist[HIDX(kpidx, i-2)] = (temphist[THIDX(kpidx, i-2)] + temphist[THIDX(kpidx, i+2)])*(1.f/16.f) +
                                     (temphist[THIDX(kpidx, i-1)] + temphist[THIDX(kpidx, i+1)])*(4.f/16.f) +
                                      temphist[THIDX(kpidx, i)]*(6.f/16.f);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Tests for multiple orientations and store keypoint
        for (int j = lid_y; j < n; j += lsz_y)
        {
            int l = j > 0 ? j - 1 : n - 1;
            int r2 = j < n-1 ? j + 1 : 0;

            //if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
            if (hist[HIDX(kpidx, j)] > hist[HIDX(kpidx, l)] &&
                hist[HIDX(kpidx, j)] > hist[HIDX(kpidx, r2)] &&
                hist[HIDX(kpidx, j)] >= mag_thr)
            {
                int finalIdx = atomic_inc(counter);

                float bin = j + 0.5f * (hist[HIDX(kpidx, l)]-hist[HIDX(kpidx, r2)]) / (hist[HIDX(kpidx, l)] - 2*hist[HIDX(kpidx, j)] + hist[HIDX(kpidx, r2)]);
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

//                keypoints_out[KOIDX(X_ROW, finalIdx)] = keypoints_in[KIIDX(X_ROW,kpidx)];
//                keypoints_out[KOIDX(Y_ROW, finalIdx)] = keypoints_in[KIIDX(Y_ROW,kpidx)];
                keypoints_out[KOIDX(X_ROW, finalIdx)] = new_real_x;
                keypoints_out[KOIDX(Y_ROW, finalIdx)] = new_real_y;
                keypoints_out[KOIDX(LAYER_ROW, finalIdx)] = layer;
                //keypoints_out[KOIDX(RESPONSE_ROW, finalIdx)] = keypoints_in[KIIDX(RESPONSE_ROW, kpidx)];
                keypoints_out[KOIDX(RESPONSE_ROW, finalIdx)] = resp;
                keypoints_out[KOIDX(OCTAVE_ROW, finalIdx)] = new_octave;
                keypoints_out[KOIDX(SIZE_ROW, finalIdx)] = new_size;
                keypoints_out[KOIDX(ANGLE_ROW, finalIdx)] = angle;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// adjustLocalExtrema

//
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
__kernel
void adjustLocalExtrema(__global const T* layer0,
                        __global const T* layer1,
                        __global const T* layer2,
                        __global const T* layer3,
                        __global const T* layer4,
                        __global const float* keypoints_in,
                        __global float* keypoints_out,
                        __global int* counter,
                        const int octave_keypoints,
                        const int max_keypoints,
                        const int octave,
                        const int n_octave_layers,
                        const float contrast_threshold,
                        const float edge_threshold,
                        const float sigma,
                        const int img_rows,
                        const int img_cols,
                        const int layer0_step,
                        const int layer1_step,
                        const int layer2_step,
                        const int layer3_step,
                        const int layer4_step,
                        const int keypoints_in_step,
                        const int keypoints_out_step)
{
    const int idx = mad24(get_group_id(0), get_local_size(0), get_local_id(0));

    if (idx < octave_keypoints)
    {

#define PIDX(A,B) mad24(A,prev_step,B)
#define CIDX(A,B) mad24(A,center_step,B)
#define NIDX(A,B) mad24(A,next_step,B)
#define KIIDX(A,B) mad24(A,keypoints_in_step,B)
#define KOIDX(A,B) mad24(A,keypoints_out_step,B)

        const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
        const float deriv_scale = img_scale*0.5f;
        const float second_deriv_scale = img_scale;
        const float cross_deriv_scale = img_scale*0.25f;

        float xi=0, xr=0, xc=0, contr=0;
        int i = 0;

        int c = keypoints_in[KIIDX(X_ROW,idx)];
        int r = keypoints_in[KIIDX(Y_ROW,idx)];
        int layer = keypoints_in[KIIDX(LAYER_ROW,idx)];

        __global const T* prev;
        __global const T* center;
        __global const T* next;
        int prev_step, center_step, next_step;

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

            float dD[3] = {(center[CIDX(r, c+1)] - center[CIDX(r, c-1)]) * deriv_scale,
                           (center[CIDX(r+1, c)] - center[CIDX(r-1, c)]) * deriv_scale,
                           (next[NIDX(r, c)] - prev[PIDX(r, c)]) * deriv_scale};

            float v2 = (float)center[CIDX(r, c)]*2.f;
            float dxx = (center[CIDX(r, c+1)] + center[CIDX(r, c-1)] - v2)*second_deriv_scale;
            float dyy = (center[CIDX(r+1, c)] + center[CIDX(r-1, c)] - v2)*second_deriv_scale;
            float dss = (next[NIDX(r, c)] + prev[PIDX(r, c)] - v2)*second_deriv_scale;
            float dxy = (center[CIDX(r+1, c+1)] - center[CIDX(r+1, c-1)] -
                         center[CIDX(r-1, c+1)] + center[CIDX(r-1, c-1)])*cross_deriv_scale;
            float dxs = (next[NIDX(r, c+1)] - next[NIDX(r, c-1)] -
                         prev[PIDX(r, c+1)] + prev[PIDX(r, c-1)])*cross_deriv_scale;
            float dys = (next[NIDX(r+1, c)] - next[NIDX(r-1, c)] -
                         prev[PIDX(r+1, c)] + prev[PIDX(r-1, c)])*cross_deriv_scale;

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

        float4 dD = {(center[CIDX(r, c+1)] - center[CIDX(r, c-1)]) * deriv_scale,
                     (center[CIDX(r+1, c)] - center[CIDX(r-1, c)]) * deriv_scale,
                     (next[NIDX(r, c)] - prev[PIDX(r, c)]) * deriv_scale,
                     0};
        float4 X = {xc, xr, xi, 0};
        float t = dot(dD, X);

        contr = center[CIDX(r, c)]*img_scale + t * 0.5f;
        if(fabs(contr) * n_octave_layers < contrast_threshold)
            return;

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = center[CIDX(r, c)]*2.f;
        float dxx = (center[CIDX(r, c+1)] + center[CIDX(r, c-1)] - v2)*second_deriv_scale;
        float dyy = (center[CIDX(r+1, c)] + center[CIDX(r-1, c)] - v2)*second_deriv_scale;
        float dxy = (center[CIDX(r+1, c+1)] - center[CIDX(r+1, c-1)] -
                     center[CIDX(r-1, c+1)] + center[CIDX(r-1, c-1)]) * cross_deriv_scale;

        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if (det <= 0 || tr*tr*edge_threshold >= (edge_threshold + 1)*(edge_threshold + 1)*det)
            return;

        int res_idx = atomic_inc(counter);

        if (res_idx < max_keypoints)
        {
            //int tmp = octave + (layer << 8) + ((int)round((xi + 0.5f)*255) << 16);
            keypoints_out[KOIDX(X_ROW,res_idx)] = (c + xc) * (1 << octave);
            keypoints_out[KOIDX(Y_ROW,res_idx)] = (r + xr) * (1 << octave);
            keypoints_out[KOIDX(RESPONSE_ROW,res_idx)] = fabs(contr);
//            keypoints_out[KOIDX(RESPONSE_ROW,res_idx)] = 1.1f;
            keypoints_out[KOIDX(OCTAVE_ROW,res_idx)] = octave;
            keypoints_out[KOIDX(LAYER_ROW,res_idx)] = layer;
            keypoints_out[KOIDX(SIZE_ROW,res_idx)] = sigma*pow(2.f, (layer + xi) / n_octave_layers)*(1 << octave)*2;
        }

#undef PIDX
#undef CIDX
#undef NIDX
#undef KIIDX
#undef KOIDX

        return;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// findExtrema

__kernel
void findExtrema(__global const T* prev,
                 __global const T* center,
                 __global const T* next,
                 __global float* keypoints,
                 __global int* counter,
                 const int octave,
                 const int scale,
                 const int max_keypoints,
                 const int threshold,
                 const int n_octave_layers,
                 const float contrast_threshold,
                 const float edge_threshold,
                 const float sigma,
                 const int img_rows,
                 const int img_cols,
                 const int prev_step,
                 const int center_step,
                 const int next_step,
                 const int keypoints_step)
{
    // One pixel border for all sides
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

#define IDX(A,B) mad24(A,mem_step,B)
#define PIDX(A,B) mad24(A,prev_step,B)
#define CIDX(A,B) mad24(A,center_step,B)
#define NIDX(A,B) mad24(A,next_step,B)
#define KIDX(A,B) mad24(A,keypoints_step,B)

    if (i < img_rows-SIFT_IMG_BORDER && j < img_cols-SIFT_IMG_BORDER)
    {
        // Copy central part
        prev_mem[IDX(y, x)] = prev[PIDX(i, j)];
        center_mem[IDX(y, x)] = center[CIDX(i, j)];
        next_mem[IDX(y, x)] = next[NIDX(i, j)];

        // Copy top border
        if (lid_i == 0)
        {
            prev_mem[IDX(y-1, x)] = prev[PIDX(i-1, j)];
            center_mem[IDX(y-1, x)] = center[CIDX(i-1, j)];
            next_mem[IDX(y-1, x)] = next[NIDX(i-1, j)];

            // Copy top left pixel
            if (lid_j == 0)
            {
                prev_mem[IDX(y-1, x-1)] = prev[PIDX(i-1, j-1)];
                center_mem[IDX(y-1, x-1)] = center[CIDX(i-1, j-1)];
                next_mem[IDX(y-1, x-1)] = next[NIDX(i-1, j-1)];
            }
            // Copy top right pixel
            else if (lid_j == lsz_j-1)
            {
                prev_mem[IDX(y-1, x+1)] = prev[PIDX(i-1, j+1)];
                center_mem[IDX(y-1, x+1)] = center[CIDX(i-1, j+1)];
                next_mem[IDX(y-1, x+1)] = next[NIDX(i-1, j+1)];
            }
        }

        // Copy left border
        if (lid_j == 0)
        {
            prev_mem[IDX(y, x-1)] = prev[PIDX(i, j-1)];
            center_mem[IDX(y, x-1)] = center[CIDX(i, j-1)];
            next_mem[IDX(y, x-1)] = next[NIDX(i, j-1)];
        }

        // Copy right border
        if (lid_j == lsz_j-1 || j == img_cols-SIFT_IMG_BORDER-1)
        {
            prev_mem[IDX(y, x+1)] = prev[PIDX(i, j+1)];
            center_mem[IDX(y, x+1)] = center[CIDX(i, j+1)];
            next_mem[IDX(y, x+1)] = next[NIDX(i, j+1)];
        }

        // Copy bottom border
        if (lid_i == lsz_i-1 || i == img_rows-SIFT_IMG_BORDER-1)
        {
            prev_mem[IDX(y+1, x)] = prev[PIDX(i+1, j)];
            center_mem[IDX(y+1, x)] = center[CIDX(i+1, j)];
            next_mem[IDX(y+1, x)] = next[NIDX(i+1, j)];

            if (lid_j == 0)
            {
                prev_mem[IDX(y+1, x-1)] = prev[PIDX(i+1, j-1)];
                center_mem[IDX(y+1, x-1)] = center[CIDX(i+1, j-1)];
                next_mem[IDX(y+1, x-1)] = next[NIDX(i+1, j-1)];
            }
            else if (lid_j == lsz_j-1 ||
                     (j == img_cols-SIFT_IMG_BORDER-1 && i == img_rows-SIFT_IMG_BORDER-1))
            {
                prev_mem[IDX(y+1, x+1)] = prev[PIDX(i+1, j+1)];
                center_mem[IDX(y+1, x+1)] = center[CIDX(i+1, j+1)];
                next_mem[IDX(y+1, x+1)] = next[NIDX(i+1, j+1)];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        T val = center_mem[IDX(y,x)];

        if (fabs((float)val) > threshold &&
            ((val > 0 && val >= center_mem[IDX(y-1, x-1)] && val >= center_mem[IDX(y-1, x)] &&
              val >= center_mem[IDX(y-1, x+1)] && val >= center_mem[IDX(y, x-1)] && val >= center_mem[IDX(y, x+1)] &&
              val >= center_mem[IDX(y+1, x-1)] && val >= center_mem[IDX(y+1, x)] && val >= center_mem[IDX(y+1, x+1)] &&
              val >= prev_mem[IDX(y-1, x-1)] && val >= prev_mem[IDX(y-1, x)] && val >= prev_mem[IDX(y-1, x+1)] &&
              val >= prev_mem[IDX(y, x-1)] && val >= prev_mem[IDX(y, x)] && val >= prev_mem[IDX(y, x+1)] &&
              val >= prev_mem[IDX(y+1, x-1)] && val >= prev_mem[IDX(y+1, x)] && val >= prev_mem[IDX(y+1, x+1)] &&
              val >= next_mem[IDX(y-1, x-1)] && val >= next_mem[IDX(y-1, x)] && val >= next_mem[IDX(y-1, x+1)] &&
              val >= next_mem[IDX(y, x-1)] && val >= next_mem[IDX(y, x)] && val >= next_mem[IDX(y, x+1)] &&
              val >= next_mem[IDX(y+1, x-1)] && val >= next_mem[IDX(y+1, x)] && val >= next_mem[IDX(y+1, x+1)]) ||
             (val < 0 && val <= center_mem[IDX(y-1, x-1)] && val <= center_mem[IDX(y-1, x)] &&
              val <= center_mem[IDX(y-1, x+1)] && val <= center_mem[IDX(y, x-1)] && val <= center_mem[IDX(y, x+1)] &&
              val <= center_mem[IDX(y+1, x-1)] && val <= center_mem[IDX(y+1, x)] && val <= center_mem[IDX(y+1, x+1)] &&
              val <= prev_mem[IDX(y-1, x-1)] && val <= prev_mem[IDX(y-1, x)] && val <= prev_mem[IDX(y-1, x+1)] &&
              val <= prev_mem[IDX(y, x-1)] && val <= prev_mem[IDX(y, x)] && val <= prev_mem[IDX(y, x+1)] &&
              val <= prev_mem[IDX(y+1, x-1)] && val <= prev_mem[IDX(y+1, x)] && val <= prev_mem[IDX(y+1, x+1)] &&
              val <= next_mem[IDX(y-1, x-1)] && val <= next_mem[IDX(y-1, x)] && val <= next_mem[IDX(y-1, x+1)] &&
              val <= next_mem[IDX(y, x-1)] && val <= next_mem[IDX(y, x)] && val <= next_mem[IDX(y, x+1)] &&
              val <= next_mem[IDX(y+1, x-1)] && val <= next_mem[IDX(y+1, x)] && val <= next_mem[IDX(y+1, x+1)])))
        {
            int idx = atomic_inc(counter);

            if (idx < max_keypoints)
            {
                keypoints[KIDX(X_ROW, idx)] = (float)j;
                keypoints[KIDX(Y_ROW, idx)] = (float)i;
                keypoints[KIDX(LAYER_ROW, idx)] = scale;
            }
            //int r1 = r, c1 = c, layer = i;
            //int r_offset = i-y, c_offset = j-x;
            //int r_mem = y, c_mem = x, layer = i;
//            int r1 = i, c1 = j, layer = scale;
//            if(!adjustLocalExtrema(prev, center, next, &layer, &r1, &c1,
//                                   n_octave_layers, contrast_threshold, edge_threshold,
//                                   sigma, img_rows, img_cols, prev_step, center_step, next_step))
//                return;

//            int r = r_offset + r_mem, c = c_offset + c_mem;
//
//            int idx = atomic_inc(counter);
//
//            if (idx < max_keypoints)
//            {
//                keypoints[KIDX(X_ROW, idx)] = c;
//                keypoints[KIDX(Y_ROW, idx)] = r;
//            }
        }
    }

#undef IDX
#undef PIDX
#undef CIDX
#undef NIDX
#undef KIDX
}
