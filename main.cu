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

struct FileStats {
	int num_docs;
	int num_terms;

	vector<int> sizes; // set sizes
	map<int, int> doc_to_class;
	vector<int> start; // beginning of each entrie

	FileStats() : num_docs(0), num_terms(0) {}
};

FileStats readInputFile(string &file, vector<Entry> &entries);
void processTestFile(InvertedIndex &index, FileStats &stats, string &file, float threshold, stringstream &fileout);
void allocIndexVariables(DeviceVariables *dev_vars, int num_terms, int block_size);

/**
 * Receives as parameters the training file name and the test file name
 */

static int num_tests = 0;
int biggestQuerySize = -1;


int main(int argc, char **argv) {

	if (argc != 5) {
		cerr << "Wrong parameters. Correct usage: <executable> <input_file> <threshold> <output_file> <number_of_gpus>" << endl;
		exit(1);
	}
	DeviceVariables dev_vars;
	vector<Entry> entries;
	InvertedIndex index;

	string inputFileName(argv[1]);
	FileStats stats = readInputFile(inputFileName, entries);

	int block_size = 100;
	int block_num = ceil((float) stats.num_docs / block_size);

	allocIndexVariables(&dev_vars, stats.num_terms, entries.size());

	for (int i = 0; i < block_num; i++) {

		int set_offset = i*block_size;
		int entries_offset = stats.start[i*block_size];
		int last_set = (i+1)*block_size >= stats.num_docs? stats.num_docs - 1: (i+1)*block_size - 1;
		int entries_size = stats.start[last_set] + stats.sizes[last_set] - entries_offset;
		printf("=========Block %d=========\nset_offset = %d\nentrie_offset: %d\nlast_set: %d\nentries_size: %d\n", i, set_offset, entries_offset, last_set, entries_size);

		index = make_inverted_index(stats.num_docs, stats.num_terms, entries_size, entries_offset, entries, &dev_vars);

		for (int j = 0; j <= i; j++) {

		}

	}

	return 0;
}

/*
 *
 *
int main(int argc, char **argv) {

	if (argc != 5) {
		cerr << "Wrong parameters. Correct usage: <executable> <input_file> <threshold> <output_file> <number_of_gpus>" << endl;
		exit(1);
	}

	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum >= atoi(argv[4])){
		gpuNum = atoi(argv[4]);
		if (gpuNum < 1)
			gpuNum = 1;
	}
	//cerr << "Using " << gpuNum << "GPUs" << endl;

	// we use 2 streams per GPU
	int numThreads = gpuNum*NUM_STREAMS;

	omp_set_num_threads(numThreads);

#if OUTPUT
	//truncate output files
	ofstream ofsf(argv[3], ofstream::trunc);
	ofsf.close();

	ofstream ofsfileoutput(argv[3], ofstream::out | ofstream::app);
#endif
	vector<InvertedIndex> indexes;
	indexes.resize(gpuNum);

	double starts, ends;

	string inputFileName(argv[1]);

	printf("Reading file...\n");
	vector<Entry> entries;

	starts = gettime();
	FileStats stats = readInputFile(inputFileName, entries);
	ends = gettime();

	printf("Time taken: %lf seconds\n", ends - starts);

	vector<stringstream*> outputString;
	//Each thread builds an output string, so it can be flushed at once at the end of the program
	for (int i = 0; i < numThreads; i++){
		outputString.push_back(new stringstream);
	}
	#pragma omp parallel num_threads(gpuNum)
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid);
		double start, end;

		start = gettime();
		indexes[cpuid] = make_inverted_index(stats.num_docs, stats.num_terms, entries.size(), entries);
		end = gettime();

		#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);
	}


	#pragma omp parallel 
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid / NUM_STREAMS);

		float threshold = atof(argv[2]);

		FileStats lstats = stats;

		processTestFile(indexes[cpuid / NUM_STREAMS], lstats, inputFileName, threshold, *outputString[cpuid]);
		if (cpuid %  NUM_STREAMS == 0)
			gpuAssert(cudaDeviceReset());

	}

#if OUTPUT
		starts = gettime();
		for (int i = 0; i < numThreads; i++){
			ofsfileoutput << outputString[i]->str();
		}
		ends = gettime();

		printf("Time taken to write output: %lf seconds\n", ends - starts);

		ofsfileoutput.close();
#endif
*/

FileStats readInputFile(string &filename, vector<Entry> &entries) {
	ifstream input(filename.c_str());
	string line;

	FileStats stats;
	int accumulatedsize = 0;
	int doc_id = 0;

	while (!input.eof()) {
		getline(input, line);
		if (line == "") continue;

		num_tests++;
		vector<string> tokens = split(line, ' ');
		biggestQuerySize = max((int)tokens.size() / 2, biggestQuerySize);

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

	stats.num_docs = num_tests;

	input.close();

	return stats;
}

void allocIndexVariables(DeviceVariables *dev_vars, int num_terms, int block_size) {
	gpuAssert(cudaMalloc(&dev_vars->d_inverted_index, block_size * sizeof(Entry)));
	gpuAssert(cudaMalloc(&dev_vars->d_entries, block_size * sizeof(Entry)));
	gpuAssert(cudaMalloc(&dev_vars->di_index, num_terms * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->di_count, num_terms * sizeof(int)));
}

void allocVariables(DeviceVariables *dev_vars, float threshold, int num_docs, Similarity** distances) {
	dim3 grid, threads;

	get_grid_config(grid, threads);

	gpuAssert(cudaMalloc(&dev_vars->d_dist, num_docs * sizeof(Similarity))); // distance between all the docs and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_result, num_docs * sizeof(Similarity))); // compacted similarities between all the docs and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_sim, num_docs * sizeof(int))); // count of elements in common
	gpuAssert(cudaMalloc(&dev_vars->d_size_doc, num_docs * sizeof(int))); // size of all docs
	gpuAssert(cudaMalloc(&dev_vars->d_query, biggestQuerySize * sizeof(Entry))); // query
	gpuAssert(cudaMalloc(&dev_vars->d_index, biggestQuerySize * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_count, biggestQuerySize * sizeof(int)));

	*distances = (Similarity*)malloc(num_docs * sizeof(Similarity));

	int blocksize = 1024;
	int numBlocks = num_docs / blocksize + (num_docs % blocksize ? 1 : 0);

	gpuAssert(cudaMalloc(&dev_vars->d_bC,sizeof(int)*(numBlocks + 1)));
	gpuAssert(cudaMalloc(&dev_vars->d_bO,sizeof(int)*numBlocks));

}

void freeVariables(DeviceVariables *dev_vars, InvertedIndex &index, Similarity** distances){
	cudaFree(dev_vars->d_dist);
	cudaFree(dev_vars->d_result);
	cudaFree(dev_vars->d_sim);
	cudaFree(dev_vars->d_size_doc);
	cudaFree(dev_vars->d_query);
	cudaFree(dev_vars->d_index);
	cudaFree(dev_vars->d_count);
	cudaFree(dev_vars->d_bC);
	cudaFree(dev_vars->d_bO);

	free(*distances);

	if (omp_get_thread_num() % NUM_STREAMS == 0){
		cudaFree(index.d_count);
		cudaFree(index.d_index);
		cudaFree(index.d_inverted_index);
		cudaFree(index.d_entries);
	}
}
