/*********************************************************************
11
12	 Copyright (C) 2015 by Wisllay Vitrio
13
14	 This program is free software; you can redistribute it and/or modify
15	 it under the terms of the GNU General Public License as published by
16	 the Free Software Foundation; either version 2 of the License, or
17	 (at your option) any later version.
18
19	 This program is distributed in the hope that it will be useful,
20	 but WITHOUT ANY WARRANTY; without even the implied warranty of
21	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
22	 GNU General Public License for more details.
23
24	 You should have received a copy of the GNU General Public License
25	 along with this program; if not, write to the Free Software
26	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
27
28	 ********************************************************************/

/* *
 * knn.cu
 */

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <set>
#include <functional>

#include "simjoin.cuh"
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "cuCompactor.cuh"


struct is_bigger_than_threshold
{
	float threshold;
	is_bigger_than_threshold(float thr) : threshold(thr) {};
	__host__ __device__
	bool operator()(const Similarity &reg)
	{
		return (reg.similarity > threshold);
	}
};

__host__ int findSimilars(InvertedIndex inverted_index, float threshold, struct DeviceVariables *dev_vars, Similarity* h_result,
		int querybegin, int queryqtt) {

	dim3 grid, threads;
	get_grid_config(grid, threads);

	int num_docs = inverted_index.num_docs;
	int *intersection = dev_vars->d_intersection, *sizes = dev_vars->d_sizes, *starts = dev_vars->d_starts;
	Entry *query = inverted_index.d_entries;
	int *compacted = dev_vars->d_compacted;
	Similarity *d_result = dev_vars->d_result;
	int *totalSimilars = (int *)malloc(sizeof(int));

	gpuAssert(cudaMemset(intersection, 0,(1 + queryqtt*num_docs)*sizeof(int)));

	calculateIntersection<<<grid, threads>>>(inverted_index, query, intersection, querybegin, queryqtt, starts, sizes, threshold);

	filter_k<<<grid, threads>>>(compacted, intersection, intersection + num_docs*queryqtt, num_docs*queryqtt, threshold, sizes, querybegin, queryqtt, num_docs);

	cudaMemcpyAsync(totalSimilars, intersection + num_docs*queryqtt, sizeof(int), cudaMemcpyDeviceToHost);

	calculateSimilarity<<<grid, threads>>>(d_result, compacted, intersection, sizes, querybegin, num_docs, queryqtt);

	cudaMemcpyAsync(h_result, d_result, totalSimilars[0]*sizeof(Similarity), cudaMemcpyDeviceToHost);

	return totalSimilars[0];
}

__global__ void calculateIntersection(InvertedIndex inverted_index, Entry *query, int *intersection, int querybegin, int queryqtt,
		int *docstart, int *docsizes, float threshold) {

	int block_start, block_end, docid, size, maxsize;

	for (int q = 0; q < queryqtt && q < inverted_index.num_docs - 1; q++) { // percorre as queries

		docid = querybegin + q;
		size = docsizes[docid];
		maxsize = ceil(((float) size)/threshold) + 1;

	for (int idx = blockIdx.x; idx < size; idx += gridDim.x) { // percorre os termos da query (apenas os que estão no midprefix)
			Entry entry = query[idx + docstart[docid]]; // find the term

			block_start = entry.term_id == 0 ? 0 : inverted_index.d_index[entry.term_id-1];
			block_end = inverted_index.d_index[entry.term_id];

			for (int i = block_start + threadIdx.x; i < block_end; i += blockDim.x) { // percorre os documentos que tem aquele termo
				Entry index_entry = inverted_index.d_inverted_index[i]; // obter item

				// somar na distância
				if (index_entry.doc_id > docid && docsizes[index_entry.doc_id] < maxsize) {
					atomicAdd(&intersection[q*inverted_index.num_docs + index_entry.doc_id], 1);
				}
			}
		}
	}
}

__global__ void filter_k (int *dst, int *src, int *nres, int n, int threshold, int *sizes, int begin, int queryqtt, int num_docs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        if (src[i] > threshold*((float) sizes[begin + i/num_docs] + sizes[i%num_docs]) / (1.0 + threshold))
            dst[atomicAdd(nres, 1)] = i;
    }
}

__global__ void calculateSimilarity(Similarity *result, int *compacted, int *intersection, int *sizes, int begin, int num_docs, int queryqtt) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < intersection[num_docs*queryqtt]; i += blockDim.x * gridDim.x) {
		result[i].doc_i = begin + compacted[i]/num_docs;
		result[i].doc_j = compacted[i]%num_docs;
		result[i].similarity = ((float) intersection[compacted[i]])/((float) sizes[result[i].doc_i] + sizes[result[i].doc_j] - intersection[compacted[i]]);
	}
}
