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

// Ordering of keypoints, descending (!IS_GT) or ascending (IS_GT)
#ifndef IS_GT
#define IS_GT false
#endif

#if IS_GT
#define COMP_OP(x,y) ((x) > (y))
#else
#define COMP_OP(x,y) ((x) < (y))
#endif

// Maximum ordering elements (limited to avoid local memory overflow)
#ifndef MAX_ELEMENTS
#define MAX_ELEMENTS 8
#endif

#define MAX_VAL_ELEMENTS MAX_ELEMENTS
#define MAX_KEY_ELEMENTS MAX_VAL_ELEMENTS

#define KIDX(Y,X) mad24(Y,keypoints_step,X)
#define KIIDX(Y,X) mad24(Y,keypoints_in_step,X)
#define KOIDX(Y,X) mad24(Y,keypoints_out_step,X)
#define COMP(A,B) keypointLessThan(A, B, key_order, key_elements)
#define COMP2(A,B) keypointLessThan2(keypoints, key_order, key_elements, A, B, keypoints_step)

static bool keypointLessThan(T* kp1,
                             T* kp2,
                             __global const int* order,
                             const int elements)
{
    for (int o = 0; o < elements; o++)
        if (kp1[o] != kp2[o])
            return COMP_OP(kp1[o], kp2[o]);

    return false;
}

static bool keypointLessThan2(__global T* keypoints,
                              __global const int* order,
                              const int elements,
                              const int kp1_idx,
                              const int kp2_idx,
                              const int keypoints_step)
{
    for (int o = 0; o < elements; o++)
        if (keypoints[KIDX(order[o], kp1_idx)] != keypoints[KIDX(order[o], kp2_idx)])
            return COMP_OP(keypoints[KIDX(order[o], kp1_idx)], keypoints[KIDX(order[o], kp2_idx)]);

    return false;
}

//  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and left and right index for a particular candidate value.  The comparison operator is
//  passed as a functor parameter my_comp
//  This function returns an index that is the first index whos value would be equal to the searched value
inline
uint lowerBoundBinary(__global T* keypoints,
                      __global const int* key_order,
                      const int key_elements,
                      const uint left,
                      const uint right,
                      const uint searchIndex,
                      const int keypoints_step)
{
    //  The values firstIndex and lastIndex get modified within the loop, narrowing down the potential sequence
    uint firstIndex = left;
    uint lastIndex = right;

//    T searchVal[MAX_KEY_ELEMENTS];
//    for (int i = 0; i < key_elements; i++)
//        searchVal[i] = keypoints[KIDX(key_order[i], (int)searchIndex)];

    //  This loops through [firstIndex, lastIndex)
    //  Since firstIndex and lastIndex will be different for every thread depending on the nested branch,
    //  this while loop will be divergent within a wavefront
    while( firstIndex < lastIndex )
    {
        //  midIndex is the average of first and last, rounded down
        uint midIndex = ( firstIndex + lastIndex ) / 2;

//        T midValue[MAX_KEY_ELEMENTS];
//        for (int i = 0; i < key_elements; i++)
//            midValue[i] = keypoints[KIDX(key_order[i], (int)midIndex)];

        //  This branch will create divergent wavefronts
        //if( my_comp( midValue, searchVal ) )
//        if( COMP( midValue, searchVal ) )
        if( COMP2( midIndex, searchIndex ) )
        {
            firstIndex = midIndex+1;
            // printf( "lowerBound: lastIndex[ %i ]=%i\n", get_local_id( 0 ), lastIndex );
        }
        else
        {
            lastIndex = midIndex;
            // printf( "lowerBound: firstIndex[ %i ]=%i\n", get_local_id( 0 ), firstIndex );
        }
    }

    return firstIndex;
}

//  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and left and right index for a particular candidate value.  The comparison operator is
//  passed as a functor parameter my_comp
//  This function returns an index that is the first index whos value would be greater than the searched value
//  If the search value is not found in the sequence, upperbound returns the same result as lowerbound
inline uint upperBoundBinary(__global T* keypoints,
                             __global const int* key_order,
                             const int key_elements,
                             const uint left,
                             const uint right,
                             const uint searchIndex,
                             const int keypoints_step)
{
    uint upperBound = lowerBoundBinary( keypoints, key_order, key_elements, left, right, searchIndex, keypoints_step );

//    T searchVal[MAX_KEY_ELEMENTS];
//    for (int i = 0; i < key_elements; i++)
//        searchVal[i] = keypoints[KIDX(key_order[i], (int)searchIndex)];

    // printf( "upperBoundBinary: upperBound[ %i, %i ]= %i\n", left, right, upperBound );
    //  If upperBound == right, then  searchVal was not found in the sequence.  Just return.
    if( upperBound != right )
    {
        //  While the values are equal i.e. !(x < y) && !(y < x) increment the index

//        T upperValue[MAX_KEY_ELEMENTS];
//        for (int i = 0; i < key_elements; i++)
//            upperValue[i] = keypoints[KIDX(key_order[i], (int)upperBound)];

//        while( !COMP( upperValue, searchVal ) && !COMP( searchVal, upperValue) && (upperBound != right) )
        while( !COMP2( upperBound, searchIndex ) && !COMP2( searchIndex, upperBound) && (upperBound != right) )
        {
            upperBound++;
//            upperValue = data[ upperBound ];
//            for (int i = 0; i < key_elements; i++)
//                upperValue[i] = keypoints[KIDX(key_order[i], (int)upperBound)];
        }
    }

    return upperBound;
}

//  This kernel implements merging of blocks of sorted data.  The input to this kernel most likely is
//  the output of blockInsertionSortTemplate.  It is expected that the source array contains multiple
//  blocks, each block is independently sorted.  The goal is to write into the output buffer half as
//  many blocks, of double the size.  The even and odd blocks are stably merged together to form
//  a new sorted block of twice the size.  The algorithm is out-of-place.
__kernel
void merge(__global T* keypoints_in,
           __global T* keypoints_out,
           __global const int* key_order,
           const uint total_keypoints,
           const uint srcLogicalBlockSize,
           const int key_elements,
           const int val_elements,
           const int keypoints_in_step,
           const int keypoints_out_step)
{
    size_t globalID     = get_global_id( 0 );

    //  Abort threads that are passed the end of the input vector
    if( globalID >= total_keypoints )
        return; // on SI this doesn't mess-up barriers

    //  For an element in sequence A, find the lowerbound index for it in sequence B
    uint srcBlockNum   = globalID / srcLogicalBlockSize;
    uint srcBlockIndex = globalID % srcLogicalBlockSize;

    // printf( "mergeTemplate: srcBlockNum[%i]=%i\n", srcBlockNum, srcBlockIndex );

    //  Pairs of even-odd blocks will be merged together
    //  An even block should search for an insertion point in the next odd block,
    //  and the odd block should look for an insertion point in the corresponding previous even block
    uint dstLogicalBlockSize = srcLogicalBlockSize<<1;
    uint leftBlockIndex = globalID & ~((dstLogicalBlockSize) - 1 );
    leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
    leftBlockIndex = min( leftBlockIndex, total_keypoints );
    uint rightBlockIndex = min( leftBlockIndex + srcLogicalBlockSize, total_keypoints );

    //  For a particular element in the input array, find the lowerbound index for it in the search sequence given by leftBlockIndex & rightBlockIndex
    // uint insertionIndex = lowerBoundLinear( iKey_ptr, leftBlockIndex, rightBlockIndex, iKey_ptr[ globalID ], my_comp ) - leftBlockIndex;
    uint insertionIndex = 0;
    if( (srcBlockNum & 0x1) == 0 )
    {
        uint lowerBound = lowerBoundBinary( keypoints_in, key_order, key_elements, leftBlockIndex, rightBlockIndex, globalID, keypoints_in_step );
        insertionIndex = lowerBound - leftBlockIndex;
    }
    else
    {
        uint upperBound = upperBoundBinary( keypoints_in, key_order, key_elements, leftBlockIndex, rightBlockIndex, globalID, keypoints_in_step );
        insertionIndex = upperBound - leftBlockIndex;
    }

    //  The index of an element in the result sequence is the summation of it's indixes in the two input
    //  sequences
    uint dstBlockIndex = srcBlockIndex + insertionIndex;
    uint dstBlockNum = srcBlockNum/2;

    uint dstIndex = (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex;
    for (int i = 0; i < val_elements; i++)
        keypoints_out[i * keypoints_out_step + dstIndex] = keypoints_in[i * keypoints_in_step + globalID];
}


__kernel
void blockInsertionSort(__global T* keypoints,
                        __global const int* key_order,
                        const uint total_keypoints,
                        const int key_elements,
                        const int val_elements,
                        const int keypoints_step)
{
    int gloId    = get_global_id( 0 );
    int groId    = get_group_id( 0 );
    int locId    = get_local_id( 0 );
    int wgSize   = get_local_size( 0 );

    bool in_range = gloId < (int)total_keypoints;

#define GIDX(A) mad24(groId,wgSize,A)

    //  Sorts a workgroup using a naive insertion sort
    //  The sort uses one thread within a workgroup to sort the entire workgroup
    if( locId == 0 && in_range )
    {
        //  The last workgroup may have an irregular size, so we calculate a per-block endIndex
        //  endIndex is essentially emulating a mod operator with subtraction and multiply
        int endIndex = (int)total_keypoints - ( groId * wgSize );
        endIndex = min( endIndex, wgSize );

        T key[MAX_KEY_ELEMENTS];
        T val[MAX_VAL_ELEMENTS];

        //  Indices are signed because the while loop will generate a -1 index inside of the max function
        for( int currIndex = 1; currIndex < endIndex; ++currIndex )
        {
            // Loads keys and values
            for (int i = 0; i < key_elements; i++)
                key[i] = keypoints[KIDX(key_order[i], GIDX(currIndex))];
            for (int i = 0; i < val_elements; i++)
                val[i] = keypoints[KIDX(i, GIDX(currIndex))];

            int scanIndex = currIndex;
            T ldsKey[MAX_KEY_ELEMENTS];
            for (int i = 0; i < key_elements; i++)
                ldsKey[i] = keypoints[KIDX(key_order[i], GIDX(scanIndex - 1))];

            while (scanIndex > 0 && COMP(key, ldsKey))
            {
                T ldsVal[MAX_VAL_ELEMENTS];
                for (int i = 0; i < val_elements; i++)
                    ldsVal[i] = keypoints[KIDX(i, GIDX(scanIndex - 1))];

                //  If the keys are being swapped, make sure the values are swapped identicaly
                for (int i = 0; i < val_elements; i++)
                    keypoints[KIDX(i, GIDX(scanIndex))] = keypoints[KIDX(i, GIDX(scanIndex - 1))];

                scanIndex = scanIndex - 1;

                for (int i = 0; i < key_elements; i++)
                    ldsKey[i] = keypoints[KIDX(key_order[i], max(0, GIDX(scanIndex - 1)))];
            }
            for (int i = 0; i < val_elements; i++)
                keypoints[KIDX(i, GIDX(scanIndex))] = val[i];
        }
    }

#undef GIDX
}

__kernel
void remove_duplicated(__global const T* keypoints_in, const int keypoints_in_step,
                       __global T* keypoints_out, const int keypoints_out_step,
                       __global const int* order, const int order_elements,
                       __global int* counter, const int total_keypoints,
                       const int keypoint_elements)
{
    const int lid_x = get_local_id(0);
    const int gid = get_global_id(0);

    if( gid >= total_keypoints )
        return;

    float fctr = 1000.f;

    // Test if all order elements are equal
    int equal_elements = 0;
#pragma unroll
    for (int i = 0; i < order_elements; i++)
        if (round(keypoints_in[KIIDX(order[i], gid)]*fctr) == round(keypoints_in[KIIDX(order[i], gid-1)]*fctr))
            equal_elements++;

    // If at least one order element is not equal, store keypoint to output matrix
    if (equal_elements != order_elements)
    {
        int idx = atomic_inc(counter);

#pragma unroll
        for( int r = 0; r < keypoint_elements; r++)
            keypoints_out[KOIDX(r,idx)] = keypoints_in[KIIDX(r,gid)];
    }
}
