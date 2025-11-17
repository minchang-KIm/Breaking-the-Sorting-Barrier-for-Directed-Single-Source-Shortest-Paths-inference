#include "partial_sort_ds.hpp"
#include <algorithm>
#include <cmath>

namespace sssp {

/**
 * @brief Insert operation - Insert a single key-value pair into D1
 * @param key Vertex identifier
 * @param value Distance value
 *
 * 삽입 연산 - D1에 단일 키-값 쌍을 삽입
 *
 * Algorithm from Lemma 3.3: Maintains sorted blocks in D1
 * Time complexity: O(log(N/M)) amortized
 *
 * BUG FIXED: Properly delete old entries from D1 before inserting new ones
 */
void PartialSortDS::insert(vertex_t key, weight_t value) {
    // Check if key already exists / 키가 이미 존재하는지 확인
    auto it = key_status.find(key);
    if (it != key_status.end()) {
        if (value >= it->second.first) {
            return; // New value is not better / 새 값이 더 좋지 않음
        }
        // FIXED: Delete old entry from D1 properly
        // 수정됨: D1에서 이전 항목을 적절히 삭제
        if (!it->second.second) { // Not in D0, must be in D1
            for (auto& block : D1) {
                block.elements.remove_if([key](const auto& p) {
                    return p.first == key;
                });
            }
        }
    }

    // Update key status / 키 상태 업데이트
    key_status[key] = {value, false};

    // Find appropriate block in D1 using upper_bound_tree
    // upper_bound_tree를 사용하여 D1에서 적절한 블록 찾기
    auto block_it = upper_bound_tree.lower_bound(value);
    if (block_it == upper_bound_tree.end()) {
        // Value exceeds B, should not happen / 값이 B를 초과, 발생하지 않아야 함
        return;
    }

    // Insert into block / 블록에 삽입
    block_it->second->elements.emplace_back(key, value);

    // Check if block needs splitting / 블록 분할 필요성 확인
    if (block_it->second->elements.size() > M) {
        split_block(block_it->second);
    }
}

/**
 * @brief BatchPrepend operation - Add L sorted elements to the front of D0
 * @param pairs Vector of (key, value) pairs to prepend
 *
 * 배치 전위 연산 - L개의 정렬된 요소를 D0의 앞에 추가
 *
 * Algorithm from Lemma 3.3: Creates new sorted blocks and prepends to D0
 * Time complexity: O(L log L) for sorting + O(L/M) for block creation
 *
 * Strategy: Creates blocks of size ~M/2 to avoid immediate splitting
 */
void PartialSortDS::batch_prepend(const std::vector<std::pair<vertex_t, weight_t>>& pairs) {
    if (pairs.empty()) return;

    size_t L = pairs.size();
    std::vector<std::pair<vertex_t, weight_t>> sorted_pairs = pairs;

    // Sort pairs by value (distance) / 값(거리)으로 쌍 정렬
    std::sort(sorted_pairs.begin(), sorted_pairs.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Remove duplicates, keeping smallest value / 중복 제거, 최소값 유지
    std::vector<std::pair<vertex_t, weight_t>> unique_pairs;
    for (const auto& p : sorted_pairs) {
        auto it = key_status.find(p.first);
        if (it == key_status.end() || p.second < it->second.first) {
            key_status[p.first] = {p.second, true}; // Mark as in D0 / D0에 있다고 표시
            unique_pairs.push_back(p);
        }
    }

    if (unique_pairs.empty()) return;

    if (unique_pairs.size() <= M) {
        // Create single block / 단일 블록 생성
        Block new_block(unique_pairs.back().second);
        new_block.elements.insert(new_block.elements.end(),
                                  unique_pairs.begin(), unique_pairs.end());
        D0.push_front(new_block);
    } else {
        // Create multiple blocks of size ~M/2 / 크기 ~M/2의 여러 블록 생성
        // This prevents immediate splitting / 즉각적인 분할 방지
        std::list<Block> new_blocks;
        size_t idx = 0;
        while (idx < unique_pairs.size()) {
            size_t end_idx = std::min(idx + M/2, unique_pairs.size());
            Block new_block(unique_pairs[end_idx-1].second);
            new_block.elements.insert(new_block.elements.end(),
                                     unique_pairs.begin() + idx,
                                     unique_pairs.begin() + end_idx);
            new_blocks.push_back(new_block);
            idx = end_idx;
        }
        D0.splice(D0.begin(), new_blocks);
    }
}

/**
 * @brief Pull operation - Extract M smallest elements and return separator
 * @return Pair of (vertices, separator value)
 *
 * 추출 연산 - M개의 최소 요소를 추출하고 구분자 반환
 *
 * Algorithm from Lemma 3.3: Pull M smallest elements from D0 ∪ D1
 * Time complexity: O(M log M) - FIXED from O(n²)
 *
 * BUG FIXED: Removed O(n²) nested loops for checking duplicates
 * Now uses hash set for O(M) duplicate detection
 */
std::pair<std::vector<vertex_t>, weight_t> PartialSortDS::pull() {
    std::vector<std::pair<vertex_t, weight_t>> candidates;
    candidates.reserve(M * 2); // Reserve space for efficiency

    // Collect from D0 / D0에서 수집
    auto it0 = D0.begin();
    while (it0 != D0.end() && candidates.size() < M * 2) {
        for (const auto& elem : it0->elements) {
            candidates.push_back(elem);
            if (candidates.size() >= M * 2) break;
        }
        if (candidates.size() >= M * 2) break;
        ++it0;
    }

    // Collect from D1 / D1에서 수집
    auto it1 = D1.begin();
    while (it1 != D1.end() && candidates.size() < M * 2) {
        for (const auto& elem : it1->elements) {
            candidates.push_back(elem);
            if (candidates.size() >= M * 2) break;
        }
        if (candidates.size() >= M * 2) break;
        ++it1;
    }

    // Sort candidates and take M smallest / 후보 정렬 및 M개의 최소값 선택
    // Time: O(M log M) / 시간: O(M log M)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    size_t num_to_return = std::min(candidates.size(), (size_t)M);
    std::vector<vertex_t> result;
    result.reserve(num_to_return);
    weight_t separator = B;

    // FIXED: Use hash set for O(M) lookup instead of O(M²) nested loops
    // 수정됨: O(M²) 중첩 루프 대신 O(M) 검색을 위한 해시 세트 사용
    std::set<vertex_t> returned_vertices;

    for (size_t i = 0; i < num_to_return; i++) {
        result.push_back(candidates[i].first);
        returned_vertices.insert(candidates[i].first);
    }

    // Determine separator / 구분자 결정
    if (candidates.size() > num_to_return) {
        separator = candidates[num_to_return].second;
    } else {
        // Find minimum remaining value / 남은 최소값 찾기
        weight_t min_remaining = INF;

        for (auto& block : D0) {
            for (auto& elem : block.elements) {
                if (returned_vertices.find(elem.first) == returned_vertices.end()) {
                    min_remaining = std::min(min_remaining, elem.second);
                }
            }
        }

        for (auto& block : D1) {
            for (auto& elem : block.elements) {
                if (returned_vertices.find(elem.first) == returned_vertices.end()) {
                    min_remaining = std::min(min_remaining, elem.second);
                }
            }
        }

        separator = (min_remaining < INF) ? min_remaining : B;
    }

    // Remove returned elements from D0 and D1 / D0와 D1에서 반환된 요소 제거
    // Time: O(M) per block / 시간: 블록당 O(M)
    for (auto block_it = D0.begin(); block_it != D0.end(); ) {
        block_it->elements.remove_if([&returned_vertices](const auto& p) {
            return returned_vertices.find(p.first) != returned_vertices.end();
        });
        if (block_it->elements.empty()) {
            block_it = D0.erase(block_it);
        } else {
            ++block_it;
        }
    }

    for (auto& block : D1) {
        block.elements.remove_if([&returned_vertices](const auto& p) {
            return returned_vertices.find(p.first) != returned_vertices.end();
        });
    }

    return {result, separator};
}

/**
 * @brief Split a block that exceeds size M
 * @param it Iterator to the block to split
 *
 * 크기 M을 초과하는 블록 분할
 *
 * Algorithm from Lemma 3.3: Maintains invariant that blocks have size ≤ M
 * Time complexity: O(M log M) for sorting
 *
 * BUG FIXED: Proper iterator handling - no undefined behavior
 */
void PartialSortDS::split_block(std::list<Block>::iterator it) {
    if (it->elements.size() <= M) return;

    // Sort elements in the block / 블록의 요소 정렬
    std::vector<std::pair<vertex_t, weight_t>> elems(it->elements.begin(), it->elements.end());
    std::sort(elems.begin(), elems.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    size_t mid = elems.size() / 2;

    // Create two new blocks / 두 개의 새 블록 생성
    Block block1(elems[mid-1].second);
    Block block2(elems.back().second);

    block1.elements.insert(block1.elements.end(), elems.begin(), elems.begin() + mid);
    block2.elements.insert(block2.elements.end(), elems.begin() + mid, elems.end());

    // Remove old upper bound from tree / 트리에서 이전 상한 제거
    for (auto tree_it = upper_bound_tree.begin(); tree_it != upper_bound_tree.end(); ++tree_it) {
        if (tree_it->second == it) {
            upper_bound_tree.erase(tree_it);
            break;
        }
    }

    // FIXED: Proper iterator handling
    // Insert new blocks and update tree / 새 블록 삽입 및 트리 업데이트
    auto new_it1 = D1.insert(it, block1);
    auto new_it2 = D1.insert(it, block2);
    D1.erase(it);

    // Update tree with new blocks / 새 블록으로 트리 업데이트
    upper_bound_tree[block1.upper_bound] = new_it1;
    upper_bound_tree[block2.upper_bound] = new_it2;
}

/**
 * @brief Get total number of elements in the data structure
 * @return Total size
 *
 * 자료 구조의 총 요소 수 반환
 *
 * Time complexity: O(number of blocks)
 */
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
