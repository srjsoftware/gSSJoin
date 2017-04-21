/*********************************************************************
11
12	 Copyright (C) 2016 by Sidney Ribeiro Junior
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

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <string>
#include <sstream>
#include <cuda.h>
#include <map>

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "simjoin.cuh"
#include "tests.cu"


#define OUTPUT 1
#define NUM_STREAMS 1

using namespace std;


FileStats readInputFile(string &file, vector<Entry> &entries);
void allocVariables(DeviceVariables *dev_vars, Pair **similar_pairs, int num_terms, int block_size, int entries_size, int num_sets);
void freeVariables(DeviceVariables *dev_vars, Pair **similar_pairs);
void write_output(Pair *similar_pairs, int totalSimilars, stringstream &outputfile);

/**
 * Receives as parameters the training file name and the test file name
 */
int main(int argc, char **argv) {

	if (argc != 6) {
		cerr << "Wrong parameters. Correct usage: <executable> <input_file> <threshold> <output_file> <number_of_gpus> <size of blocks>" << endl;
		exit(1);
	}

	vector<Entry> entries;
	float threshold = atof(argv[2]);
	int gpuNum;

	cudaGetDeviceCount(&gpuNum);
	if (gpuNum > atoi(argv[4]) && atoi(argv[4]) > 0)
		gpuNum = atoi(argv[4]);
	omp_set_num_threads(gpuNum);

	string inputFileName(argv[1]);
	printf("Reading file %s...\n", inputFileName.c_str());
	FileStats stats = readInputFile(inputFileName, entries);

	ofstream ofsf(argv[3], ofstream::trunc);
	ofsf.close();
	ofstream ofsfileoutput(argv[3], ofstream::out | ofstream::app);
	vector<stringstream*> outputString; // Each thread has an output string.
	for (int i = 0; i < gpuNum; i++)
		outputString.push_back(new stringstream);

	// calculating maximum size of data structures
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	long sizeEntries = (stats.start[stats.num_sets - 1] + stats.sizes[stats.num_sets - 1]) * sizeof(Entry);
	long sizeVectorsN = stats.num_sets*sizeof(int);
	long freeMem = free_mem - 3*sizeEntries - sizeVectorsN*4;

	int block_size = atoi(argv[5]);
	block_size = block_size < 1? freeMem / (stats.num_sets*(sizeof(float) + sizeof(Pair))): block_size;
	block_size = block_size > stats.num_sets? stats.num_sets: block_size;
	int block_num = ceil((float) stats.num_sets / block_size);

	double start = gettime();

	#pragma omp parallel num_threads(gpuNum)
	{
		int gpuid = omp_get_thread_num();
		cudaSetDevice(gpuid);
		InvertedIndex index;
		DeviceVariables dev_vars;
		Pair *similar_pairs;

		allocVariables(&dev_vars, &similar_pairs, stats.num_terms, block_size, entries.size(), stats.num_sets);
		gpuAssert(cudaMemcpy(dev_vars.d_starts, &stats.start[0], stats.num_sets * sizeof(int), cudaMemcpyHostToDevice));
		gpuAssert(cudaMemcpy(dev_vars.d_sizes, &stats.sizes[0], stats.num_sets * sizeof(int), cudaMemcpyHostToDevice));

		for (int i = gpuid; i < block_num; i+= gpuNum) {
			int entries_block_start = i*block_size;
			int entries_offset = stats.start[entries_block_start];
			int last_set = entries_block_start + block_size >= stats.num_sets? stats.num_sets - 1: entries_block_start + block_size - 1;
			int entries_block_size = last_set - entries_block_start + 1;
			int entries_size = stats.start[last_set] + stats.sizes[last_set] - entries_offset;
			//printf("=========Indexed Block %d=========\nset_offset = %d\nentrie_offset: %d\nlast_set: %d\nentries_size: %d\n", i, entries_block_start, entries_offset, last_set, entries_size);

			// build the inverted index for block i of size block_size
			index = make_inverted_index(stats.num_sets, stats.num_terms, entries_size, entries_offset, entries, &dev_vars);
			//print_sets(entries, stats.sizes, stats.start); //print_invertedIndex(index);

			for (int j = 0; j <= i; j++) { // calculate similarity between indexed sets and probe sets
				int probe_block_start = j*block_size;
				int last_probe = probe_block_start + block_size > stats.num_sets? stats.num_sets - 1: probe_block_start + block_size - 1;
				int probe_block_size = last_probe - probe_block_start + 1;
				int probes_offset = stats.start[probe_block_start];

				// size filtering
				if (stats.sizes[last_probe] < threshold * stats.sizes[entries_block_start])
					continue;

				if (j < i) {
					int probes_size = stats.start[last_probe] + stats.sizes[last_probe] - probes_offset;
					gpuAssert(cudaMemcpy(dev_vars.d_probes, &entries[probes_offset], probes_size * sizeof(Entry), cudaMemcpyHostToDevice));
				}
				//printf("=========Probe Block %d=========\nprobe_block_start = %d\nprobe_offset: %d\nlast_probe: %d\nprobe_block_size: %d\n===============================\n", j, probe_block_start, probes_offset,last_probe, probe_block_size);

				int totalSimilars = findSimilars(index, threshold, &dev_vars, similar_pairs, probe_block_start,
						probe_block_size, probes_offset, entries_block_size, entries_block_start, i, j);

				//print_intersection(dev_vars.d_intersection, block_size, i, j);
				//print_result(similar_pairs, totalSimilars);
				write_output(similar_pairs, totalSimilars, *outputString[gpuid]);
			}

		}

		freeVariables(&dev_vars, &similar_pairs);
	}

	double end = gettime();

	printf("Time to process similarity join between %d sets: %lf seconds\n", stats.num_sets, end - start);

	for (int i = 0; i < gpuNum; i++)
		ofsfileoutput << outputString[i]->str();
	ofsfileoutput.close();

	return 0;
}

FileStats readInputFile(string &filename, vector<Entry> &entries) {
	ifstream input(filename.c_str());
	string line;

	FileStats stats;
	int accumulatedsize = 0;
	int doc_id = 0;

	while (!input.eof()) {
		getline(input, line);
		if (line == "") continue;

		vector<string> tokens = split(line, ' ');
		//biggestQuerySize = max((int)tokens.size() / 2, biggestQuerySize);

		int size = (tokens.size() - 2)/2;
		stats.sizes.push_back(size);
		stats.start.push_back(accumulatedsize);
		accumulatedsize += size;

		for (int i = 2, size = tokens.size(); i + 1 < size; i += 2) {
			int term_id = atoi(tokens[i].c_str());
			int term_count = atoi(tokens[i + 1].c_str());
			stats.num_terms = max(stats.num_terms, term_id + 1);
			entries.push_back(Entry(doc_id, term_id, term_count));
		}
		doc_id++;
	}

	stats.num_sets = stats.start.size();

	input.close();

	return stats;
}

void allocVariables(DeviceVariables *dev_vars, Pair **similar_pairs, int num_terms, int block_size, int entries_size, int num_sets) {
	// TODO alocar o tamanho certo para entries, probes e o índice invertido

	// Inverted index's variables
	gpuAssert(cudaMalloc(&dev_vars->d_inverted_index, entries_size * sizeof(Entry)));
	gpuAssert(cudaMalloc(&dev_vars->d_entries, entries_size * sizeof(Entry)));
	gpuAssert(cudaMalloc(&dev_vars->d_index, num_terms * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_count, num_terms * sizeof(int)));

	// Variables used to perform the similarity join
	gpuAssert(cudaMalloc(&dev_vars->d_probes, entries_size * sizeof(Entry)));
	gpuAssert(cudaMalloc(&dev_vars->d_intersection, (1 + block_size * block_size) * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_pairs, block_size *block_size * sizeof(Pair)));
	gpuAssert(cudaMalloc(&dev_vars->d_sizes, num_sets * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_starts, num_sets * sizeof(int)));

	*similar_pairs = (Pair *)malloc(sizeof(Pair)*block_size*block_size);
}

void freeVariables(DeviceVariables *dev_vars, Pair **similar_pairs) {
	cudaFree(&dev_vars->d_inverted_index);
	cudaFree(&dev_vars->d_entries);
	cudaFree(&dev_vars->d_index);
	cudaFree(&dev_vars->d_count);

	cudaFree(&dev_vars->d_probes);
	cudaFree(&dev_vars->d_intersection);
	cudaFree(&dev_vars->d_pairs);
	cudaFree(&dev_vars->d_sizes);
	cudaFree(&dev_vars->d_starts);

	free(*similar_pairs);
}

void write_output(Pair *similar_pairs, int totalSimilars, stringstream &outputfile) {
	for (int i = 0; i < totalSimilars; i++) {
		outputfile << "(" << similar_pairs[i].set_x << ", " << similar_pairs[i].set_y << "): " << similar_pairs[i].similarity << endl;
	}
}
