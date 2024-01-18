#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>

namespace {

template <typename T>
T sign(T x) {
    if (x > 0) {
        return T(1);
    }
    if (x < 0) {
        return T(-1);
    }
    return T(0);
}

//! pair of T and int; compare only on T
template <typename T>
struct TIpair {
    T t;
    int i;

    TIpair() = default;
    constexpr TIpair(T t, int i) : t{t}, i{i} {}

    bool operator<(const TIpair& other) const { return t < other.t; }
};

template <typename T>
struct SlackItem {
    T cost;  //! cost of slack
    int sum_id, comp_id;

    SlackItem() = default;
    SlackItem(T cost, int sum_id, int comp_id)
            : cost{cost}, sum_id{sum_id}, comp_id{comp_id} {}

    //! reverse compare to be used with priority_queue
    bool operator<(const SlackItem& other) const { return cost > other.cost; }
};

template <typename T, int N>
struct SmallArray {
    T data[N];
    uint32_t head;
};

/*!
 * compute the mask of activated components of the function
 * sum_{i=0}^{comp_size-1} max_{j=0}^{NMAX-1} comp[i*NMAX+j]
 *
 * \param[out] mask bitmask of activated components for each summand; does
 *      not need to be initialized
 */
template <int NMAX, typename T = double>
void compute_sum_of_max_subd_mask(T tot_slack, uint8_t* mask, const T* comp,
                                  size_t comp_size) {
    static_assert(NMAX >= 2 && NMAX <= 8, "NMAX must be in [2, 8]");

    std::vector<SlackItem<T>> slack_init;
    std::vector<SmallArray<TIpair<T>, NMAX - 1>> comp_sorted;
    comp_sorted.reserve(comp_size);

    for (size_t sum_i = 0; sum_i < comp_size; ++sum_i) {
        TIpair<T> cval[NMAX];

        // sort cval in decreasing order
        for (int i = 0; i < NMAX; ++i) {
            cval[i] = {comp[sum_i * NMAX + i], i};
            for (int j = i; j > 0; --j) {
                bool swap = cval[j - 1] < cval[j];
                TIpair<T> tmp = cval[j];
                cval[j] = swap ? cval[j - 1] : tmp;
                cval[j - 1] = swap ? tmp : cval[j - 1];
            }
        }

        mask[sum_i] = 1 << cval[0].i;
        slack_init.emplace_back(cval[0].t - cval[1].t, sum_i, cval[1].i);

        auto& st = comp_sorted.emplace_back();
        for (int i = 1; i < NMAX; ++i) {
            st.data[i - 1] = cval[i];
        }
        st.head = 0;
    }

    std::priority_queue<SlackItem<T>> slack(slack_init.begin(),
                                            slack_init.end());
    while (tot_slack >= 0 && !slack.empty()) {
        SlackItem<T> top = slack.top();
        slack.pop();
        if (tot_slack < top.cost) {
            break;
        }
        tot_slack -= top.cost;
        mask[top.sum_id] |= 1 << top.comp_id;

        auto& st = comp_sorted[top.sum_id];
        if (st.head < NMAX - 2) {
            slack.emplace(st.data[st.head].t - st.data[st.head + 1].t,
                          top.sum_id, st.data[st.head + 1].i);
            ++st.head;
        }
    }
}

}  // anonymous namespace
