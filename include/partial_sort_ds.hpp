#ifndef PARTIAL_SORT_DS_HPP
#define PARTIAL_SORT_DS_HPP

#include "graph.hpp"
#include <list>
#include <set>
#include <map>
#include <algorithm>

namespace sssp {

// Block-based linked list data structure from Lemma 3.3
class PartialSortDS {
private:
    struct Block {
        std::list<std::pair<vertex_t, weight_t>> elements;
        weight_t upper_bound;

        Block(weight_t ub = INF) : upper_bound(ub) {}
    };

    std::list<Block> D0; // Blocks from BatchPrepend
    std::list<Block> D1; // Blocks from Insert
    std::map<weight_t, std::list<Block>::iterator> upper_bound_tree; // For D1
    std::map<vertex_t, std::pair<weight_t, bool>> key_status; // Track key status (value, is_in_D0)

    uint32_t M; // Block size parameter
    weight_t B; // Upper bound on all values
    uint32_t N; // Maximum number of insertions

    void split_block(std::list<Block>::iterator it);
    weight_t find_median(std::list<std::pair<vertex_t, weight_t>>& elements);

public:
    PartialSortDS(uint32_t max_insertions, uint32_t block_size, weight_t upper_bound)
        : M(block_size), B(upper_bound), N(max_insertions) {
        // Initialize D1 with one empty block
        D1.emplace_back(B);
        upper_bound_tree[B] = D1.begin();
    }

    // Insert a key/value pair
    void insert(vertex_t key, weight_t value);

    // Batch prepend multiple key/value pairs
    void batch_prepend(const std::vector<std::pair<vertex_t, weight_t>>& pairs);

    // Pull M smallest elements
    std::pair<std::vector<vertex_t>, weight_t> pull();

    // Check if empty
    bool empty() const {
        return D0.empty() && (D1.size() == 1 && D1.front().elements.empty());
    }

    size_t size() const;
};

} // namespace sssp

#endif // PARTIAL_SORT_DS_HPP
