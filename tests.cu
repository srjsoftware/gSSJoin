#include <stdio.h>
#include <stdlib.h>

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"

void print_invertedIndex(InvertedIndex index) {
	printf("Docs: %d\nEntries: %d\nTerms: %d\n", index.num_docs, index.num_entries, index.num_terms);

	Entry *inverted_index = (Entry*)malloc(sizeof(Entry)*index.num_entries);
	cudaMemcpyAsync(inverted_index, index.d_inverted_index, sizeof(Entry)*index.num_entries, cudaMemcpyDeviceToHost);

	int *count = (int *)malloc(sizeof(int)*index.num_terms);
	cudaMemcpyAsync(count, index.d_count, sizeof(int)*index.num_terms, cudaMemcpyDeviceToHost);

	int *h_index = (int *)malloc(sizeof(int)*index.num_terms);
	cudaMemcpyAsync(h_index, index.d_index, sizeof(int)*index.num_terms, cudaMemcpyDeviceToHost);

	printf("Count: ");
	for (int i = 0; i < index.num_terms; i++) {
		printf("%d ", count[i]);
	}

	printf("\nList's ends: ");
		for (int i = 0; i < index.num_terms; i++) {
			printf("%d ", h_index[i]);
	}

	printf("\nIndex:");
	int term = -1;
	for (int i = 0; i < index.num_entries; i++) {
		if (term != inverted_index[i].term_id) {
			printf("\n[%d]: ", inverted_index[i].term_id);
			term = inverted_index[i].term_id;
		}
		printf("%d ", inverted_index[i].doc_id);
	}
	printf("\n");
}
