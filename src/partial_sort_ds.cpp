#include "partial_sort_ds.hpp"
#include <algorithm>
#include <cmath>

namespace sssp {

void PartialSortDS::insert(vertex_t key, weight_t value) {
    // Check if key already exists
    auto it = key_status.find(key);
    if (it != key_status.end()) {
        if (value >= it->second.first) {
            return; // New value is not better
        }
        // Delete old entry (we'll insert new one below)
        // This is simplified - in practice, we'd need to track and delete from D0/D1
    }

    // Update key status
    key_status[key] = {value, false};

    // Find appropriate block in D1 using upper_bound_tree
    auto block_it = upper_bound_tree.lower_bound(value);
    if (block_it == upper_bound_tree.end()) {
        // Should not happen if B is set correctly
        return;
    }

    // Insert into block
    block_it->second->elements.emplace_back(key, value);

    // Check if block needs splitting
    if (block_it->second->elements.size() > M) {
        split_block(block_it->second);
    }
}

void PartialSortDS::batch_prepend(const std::vector<std::pair<vertex_t, weight_t>>& pairs) {
    if (pairs.empty()) return;

    size_t L = pairs.size();
    std::vector<std::pair<vertex_t, weight_t>> sorted_pairs = pairs;
    std::sort(sorted_pairs.begin(), sorted_pairs.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Remove duplicates, keeping smallest value
    for (const auto& p : sorted_pairs) {
        auto it = key_status.find(p.first);
        if (it == key_status.end() || p.second < it->second.first) {
            key_status[p.first] = {p.second, true};
        }
    }

    if (L <= M) {
        // Create single block
        Block new_block(sorted_pairs.back().second);
        new_block.elements.insert(new_block.elements.end(),
                                  sorted_pairs.begin(), sorted_pairs.end());
        D0.push_front(new_block);
    } else {
        // Create multiple blocks of size ~M/2
        std::list<Block> new_blocks;
        size_t idx = 0;
        while (idx < L) {
            size_t end_idx = std::min(idx + M/2, L);
            Block new_block(sorted_pairs[end_idx-1].second);
            new_block.elements.insert(new_block.elements.end(),
                                     sorted_pairs.begin() + idx,
                                     sorted_pairs.begin() + end_idx);
            new_blocks.push_back(new_block);
            idx = end_idx;
        }
        D0.splice(D0.begin(), new_blocks);
    }
}

std::pair<std::vector<vertex_t>, weight_t> PartialSortDS::pull() {
    std::vector<std::pair<vertex_t, weight_t>> candidates;

    // Collect from D0
    auto it0 = D0.begin();
    while (it0 != D0.end() && candidates.size() < M) {
        for (const auto& elem : it0->elements) {
            candidates.push_back(elem);
            if (candidates.size() >= M) break;
        }
        if (candidates.size() >= M) break;
        ++it0;
    }

    // Collect from D1
    auto it1 = D1.begin();
    while (it1 != D1.end() && candidates.size() < M) {
        for (const auto& elem : it1->elements) {
            candidates.push_back(elem);
            if (candidates.size() >= M) break;
        }
        if (candidates.size() >= M) break;
        ++it1;
    }

    // Sort candidates and take M smallest
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    size_t num_to_return = std::min(candidates.size(), (size_t)M);
    std::vector<vertex_t> result;
    weight_t separator = B;

    for (size_t i = 0; i < num_to_return; i++) {
        result.push_back(candidates[i].first);
    }

    // Determine separator
    if (candidates.size() > num_to_return) {
        separator = candidates[num_to_return].second;
    } else if (!D0.empty() || !D1.front().elements.empty()) {
        // Find minimum remaining value
        weight_t min_remaining = INF;

        for (auto& block : D0) {
            for (auto& elem : block.elements) {
                bool found = false;
                for (size_t i = 0; i < num_to_return; i++) {
                    if (candidates[i].first == elem.first) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    min_remaining = std::min(min_remaining, elem.second);
                }
            }
        }

        for (auto& block : D1) {
            for (auto& elem : block.elements) {
                bool found = false;
                for (size_t i = 0; i < num_to_return; i++) {
                    if (candidates[i].first == elem.first) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    min_remaining = std::min(min_remaining, elem.second);
                }
            }
        }

        separator = min_remaining;
    }

    // Remove returned elements from D0 and D1
    for (const auto& v : result) {
        // Remove from D0
        for (auto block_it = D0.begin(); block_it != D0.end(); ) {
            block_it->elements.remove_if([v](const auto& p) { return p.first == v; });
            if (block_it->elements.empty()) {
                block_it = D0.erase(block_it);
            } else {
                ++block_it;
            }
        }

        // Remove from D1
        for (auto block_it = D1.begin(); block_it != D1.end(); ) {
            block_it->elements.remove_if([v](const auto& p) { return p.first == v; });
            ++block_it;
        }
    }

    return {result, separator};
}

void PartialSortDS::split_block(std::list<Block>::iterator it) {
    if (it->elements.size() <= M) return;

    std::vector<std::pair<vertex_t, weight_t>> elems(it->elements.begin(), it->elements.end());
    std::sort(elems.begin(), elems.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    size_t mid = elems.size() / 2;

    Block block1(elems[mid-1].second);
    Block block2(elems.back().second);

    block1.elements.insert(block1.elements.end(), elems.begin(), elems.begin() + mid);
    block2.elements.insert(block2.elements.end(), elems.begin() + mid, elems.end());

    // Remove old upper bound from tree
    for (auto tree_it = upper_bound_tree.begin(); tree_it != upper_bound_tree.end(); ++tree_it) {
        if (tree_it->second == it) {
            upper_bound_tree.erase(tree_it);
            break;
        }
    }

    // Insert new blocks
    auto new_it = D1.insert(it, block1);
    D1.insert(it, block2);
    D1.erase(it);

    // Update tree
    upper_bound_tree[block1.upper_bound] = new_it;
    upper_bound_tree[block2.upper_bound] = ++new_it;
}

size_t PartialSortDS::size() const {
    size_t total = 0;
    for (const auto& block : D0) {
        total += block.elements.size();
    }
    for (const auto& block : D1) {
        total += block.elements.size();
    }
    return total;
}

} // namespace sssp
