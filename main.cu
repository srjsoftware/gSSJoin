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

	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > atoi(argv[4])){
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
	vector<string> inputs;// to read the whole test file in memory
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

	//create an inverted index for all streams in each GPU
	#pragma omp parallel num_threads(gpuNum)
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid+1);
		double start, end;

		start = gettime();
		indexes[cpuid] = make_inverted_index(stats.num_docs, stats.num_terms, entries);
		end = gettime();

		#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);
	}


	#pragma omp parallel 
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid + 1 / NUM_STREAMS);

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

void allocVariables(DeviceVariables *dev_vars, float threshold, int num_docs, Similarity** h_result, int queryqtt){
	dim3 grid, threads;

	get_grid_config(grid, threads);

	gpuAssert(cudaMalloc(&dev_vars->d_result, queryqtt * num_docs * sizeof(Similarity))); // compacted similarities between all the docs and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_intersection, sizeof(int) + queryqtt * num_docs * sizeof(int))); // count of elements in common
	gpuAssert(cudaMalloc(&dev_vars->d_similarity, queryqtt * num_docs * sizeof(Similarity))); // similarity between all the docs and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_sizes, num_docs * sizeof(int))); // size of all docs
	gpuAssert(cudaMalloc(&dev_vars->d_starts, num_docs * sizeof(int))); // size of all docs

	*h_result = (Similarity*)malloc(queryqtt * num_docs * sizeof(Similarity));

	int blocksize = 1024;
	int numBlocks = (num_docs*queryqtt) / blocksize + ((num_docs*queryqtt) % blocksize ? 1 : 0);

	gpuAssert(cudaMalloc(&dev_vars->d_bC,sizeof(int)*(numBlocks + 1)));
	gpuAssert(cudaMalloc(&dev_vars->d_bO,sizeof(int)*numBlocks));
}

void freeVariables(DeviceVariables *dev_vars, InvertedIndex &index, Similarity** h_result){
	cudaFree(dev_vars->d_result);
	cudaFree(dev_vars->d_intersection);
	cudaFree(dev_vars->d_similarity);
	cudaFree(dev_vars->d_sizes);
	cudaFree(dev_vars->d_starts);
	cudaFree(dev_vars->d_bC);
	cudaFree(dev_vars->d_bO);

	free(*h_result);

	if (omp_get_thread_num() % NUM_STREAMS == 0){
		cudaFree(index.d_count);
		cudaFree(index.d_index);
		cudaFree(index.d_inverted_index);
		cudaFree(index.d_entries);
	}
}

void processTestFile(InvertedIndex &index, FileStats &stats, string &filename, float threshold, stringstream &outputfile) {

	int num_test_local = 0, querybegin;

	//#pragma omp single nowait
	printf("Processing input file %s...\n", filename.c_str());

	DeviceVariables dev_vars;
	Similarity* h_result;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	long gpuGlobalMem = deviceProp.totalGlobalMem;
	long sizeEntries = (stats.start[stats.num_docs - 1] + stats.sizes[stats.num_docs - 1]) * sizeof(Entry);
	long sizeVectorsSizeAndStart = 2*stats.num_docs*sizeof(int);
	long sizeOfInvertedIndex = sizeEntries + 2*stats.num_terms*sizeof(int);
	long freeMem = gpuGlobalMem - sizeEntries - sizeVectorsSizeAndStart - sizeOfInvertedIndex;

	int queryqtt = freeMem / (stats.num_docs*(2*sizeof(int) + 2*sizeof(Similarity)));
	queryqtt = queryqtt > stats.num_docs? stats.num_docs: queryqtt;

	allocVariables(&dev_vars, threshold, index.num_docs, &h_result, queryqtt);

	cudaMemcpyAsync(dev_vars.d_sizes, &stats.sizes[0], index.num_docs * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_vars.d_starts, &stats.start[0], index.num_docs * sizeof(int), cudaMemcpyHostToDevice);

	double start = gettime();

#pragma omp for
	for (querybegin = 0; querybegin < index.num_docs - 1; querybegin += queryqtt) {

		num_test_local++;

		int totalSimilars = findSimilars(index, threshold, &dev_vars, h_result, querybegin, queryqtt);

		for (int i = 0; i < totalSimilars; i++) {
#if OUTPUT
			outputfile << "(" << h_result[i].doc_i << ", " << h_result[i].doc_j << "): " << h_result[i].similarity << endl;
#endif
		}
	}

	freeVariables(&dev_vars, index, &h_result);
	int threadid = omp_get_thread_num();

	printf("Entries in device %d stream %d: %d\n", threadid / NUM_STREAMS, threadid %  NUM_STREAMS, num_test_local);

	#pragma omp barrier

	double end = gettime();

	#pragma omp master
	{
		printf("Time taken for %d queries: %lf seconds\n\n", num_tests, end - start);
	}

}
