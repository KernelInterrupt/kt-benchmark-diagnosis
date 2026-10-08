#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr double kLogHalf = -0.693147180559945309417232121458176568;
thread_local std::string g_last_error;

double logsumexp_pair(double a, double b) {
    if (a > b) {
        return a + std::log1p(std::exp(b - a));
    }
    return b + std::log1p(std::exp(a - b));
}

struct Node {
    int count0 = 0;
    int count1 = 0;
    double log_pe = 0.0;
    double log_pw = 0.0;
    double child_log_pw_sum = 0.0;
    std::unordered_map<int, std::unique_ptr<Node>> children;

    Node() = default;

    Node(const Node& other)
        : count0(other.count0),
          count1(other.count1),
          log_pe(other.log_pe),
          log_pw(other.log_pw),
          child_log_pw_sum(other.child_log_pw_sum) {
        children.reserve(other.children.size());
        for (const auto& kv : other.children) {
            children.emplace(kv.first, std::make_unique<Node>(*kv.second));
        }
    }

    Node& operator=(const Node& other) {
        if (this == &other) {
            return *this;
        }
        count0 = other.count0;
        count1 = other.count1;
        log_pe = other.log_pe;
        log_pw = other.log_pw;
        child_log_pw_sum = other.child_log_pw_sum;
        children.clear();
        children.reserve(other.children.size());
        for (const auto& kv : other.children) {
            children.emplace(kv.first, std::make_unique<Node>(*kv.second));
        }
        return *this;
    }

    double local_predictive_prob(int target) const {
        const int total = count0 + count1;
        if (target == 1) {
            return (static_cast<double>(count1) + 0.5) / (static_cast<double>(total) + 1.0);
        }
        return (static_cast<double>(count0) + 0.5) / (static_cast<double>(total) + 1.0);
    }

    void recompute_log_pw(int depth_remaining) {
        if (depth_remaining <= 0 || children.empty()) {
            log_pw = log_pe;
            return;
        }
        log_pw = logsumexp_pair(kLogHalf + log_pe, kLogHalf + child_log_pw_sum);
    }

    void update(const int* context_suffix_rev, int len, int target, int depth_remaining) {
        log_pe += std::log(local_predictive_prob(target));
        if (target == 1) {
            count1 += 1;
        } else {
            count0 += 1;
        }

        if (depth_remaining > 0 && len > 0) {
            const int token = context_suffix_rev[0];
            auto it = children.find(token);
            double old_child_log_pw = 0.0;
            if (it != children.end()) {
                old_child_log_pw = it->second->log_pw;
            } else {
                auto inserted = children.emplace(token, std::make_unique<Node>());
                it = inserted.first;
            }
            it->second->update(context_suffix_rev + 1, len - 1, target, depth_remaining - 1);
            child_log_pw_sum += it->second->log_pw - old_child_log_pw;
        }

        recompute_log_pw(depth_remaining);
    }

    double hypothetical_log_pw(const int* context_suffix_rev, int len, int target, int depth_remaining) const;
};

double empty_hypothetical_log_pw(const int* context_suffix_rev, int len, int target, int depth_remaining) {
    const double log_pe_after = std::log(0.5);
    if (depth_remaining <= 0 || len == 0) {
        return log_pe_after;
    }
    const double child_after =
        empty_hypothetical_log_pw(context_suffix_rev + 1, len - 1, target, depth_remaining - 1);
    return logsumexp_pair(kLogHalf + log_pe_after, kLogHalf + child_after);
}

double Node::hypothetical_log_pw(
    const int* context_suffix_rev,
    int len,
    int target,
    int depth_remaining
) const {
    const double log_pe_after = log_pe + std::log(local_predictive_prob(target));
    const bool has_child_extension = depth_remaining > 0 && len > 0;
    const bool has_children_after = !children.empty() || has_child_extension;
    if (depth_remaining <= 0 || !has_children_after) {
        return log_pe_after;
    }

    double child_log_pw_sum_after = child_log_pw_sum;
    if (has_child_extension) {
        const int token = context_suffix_rev[0];
        auto it = children.find(token);
        const double old_child_log_pw = (it != children.end()) ? it->second->log_pw : 0.0;
        double child_log_pw_after = 0.0;
        if (it == children.end()) {
            child_log_pw_after =
                empty_hypothetical_log_pw(context_suffix_rev + 1, len - 1, target, depth_remaining - 1);
        } else {
            child_log_pw_after =
                it->second->hypothetical_log_pw(context_suffix_rev + 1, len - 1, target, depth_remaining - 1);
        }
        child_log_pw_sum_after += child_log_pw_after - old_child_log_pw;
    }

    return logsumexp_pair(kLogHalf + log_pe_after, kLogHalf + child_log_pw_sum_after);
}

struct CtwEstimator {
    explicit CtwEstimator(int depth) : max_depth(depth) {}
    int max_depth;
    Node root;
    struct NodeStateUndo {
        Node* node = nullptr;
        int count0 = 0;
        int count1 = 0;
        double log_pe = 0.0;
        double log_pw = 0.0;
        double child_log_pw_sum = 0.0;
    };
    struct ChildInsertUndo {
        Node* parent = nullptr;
        int token = 0;
    };
    struct UndoEntry {
        bool is_insert = false;
        NodeStateUndo state;
        ChildInsertUndo insert;
    };
    std::vector<UndoEntry> undo_log;
};

bool validate_context(const int* tokens, int len) {
    return (len == 0) || (tokens != nullptr);
}

void push_state_undo(CtwEstimator& estimator, Node* node) {
    CtwEstimator::UndoEntry entry;
    entry.is_insert = false;
    entry.state.node = node;
    entry.state.count0 = node->count0;
    entry.state.count1 = node->count1;
    entry.state.log_pe = node->log_pe;
    entry.state.log_pw = node->log_pw;
    entry.state.child_log_pw_sum = node->child_log_pw_sum;
    estimator.undo_log.push_back(entry);
}

void push_insert_undo(CtwEstimator& estimator, Node* parent, int token) {
    CtwEstimator::UndoEntry entry;
    entry.is_insert = true;
    entry.insert.parent = parent;
    entry.insert.token = token;
    estimator.undo_log.push_back(entry);
}

void update_with_undo(
    CtwEstimator& estimator,
    Node* node,
    const int* context_suffix_rev,
    int len,
    int target,
    int depth_remaining
) {
    push_state_undo(estimator, node);
    node->log_pe += std::log(node->local_predictive_prob(target));
    if (target == 1) {
        node->count1 += 1;
    } else {
        node->count0 += 1;
    }

    if (depth_remaining > 0 && len > 0) {
        const int token = context_suffix_rev[0];
        auto it = node->children.find(token);
        double old_child_log_pw = 0.0;
        if (it != node->children.end()) {
            old_child_log_pw = it->second->log_pw;
        } else {
            push_insert_undo(estimator, node, token);
            auto inserted = node->children.emplace(token, std::make_unique<Node>());
            it = inserted.first;
        }
        update_with_undo(estimator, it->second.get(), context_suffix_rev + 1, len - 1, target, depth_remaining - 1);
        node->child_log_pw_sum += it->second->log_pw - old_child_log_pw;
    }

    node->recompute_log_pw(depth_remaining);
}

void rollback_to(CtwEstimator& estimator, std::size_t marker) {
    while (estimator.undo_log.size() > marker) {
        auto entry = estimator.undo_log.back();
        estimator.undo_log.pop_back();
        if (entry.is_insert) {
            auto* parent = entry.insert.parent;
            if (parent != nullptr) {
                parent->children.erase(entry.insert.token);
            }
        } else {
            auto* node = entry.state.node;
            if (node != nullptr) {
                node->count0 = entry.state.count0;
                node->count1 = entry.state.count1;
                node->log_pe = entry.state.log_pe;
                node->log_pw = entry.state.log_pw;
                node->child_log_pw_sum = entry.state.child_log_pw_sum;
            }
        }
    }
}

}  // namespace

extern "C" {

struct CtwHandle {
    CtwEstimator estimator;
    explicit CtwHandle(int depth) : estimator(depth) {}
};

const char* ctw_last_error() {
    return g_last_error.c_str();
}

CtwHandle* ctw_create(int max_depth) {
    try {
        if (max_depth < 1) {
            g_last_error = "max_depth must be at least 1";
            return nullptr;
        }
        return new CtwHandle(max_depth);
    } catch (const std::exception& ex) {
        g_last_error = ex.what();
        return nullptr;
    }
}

void ctw_destroy(CtwHandle* handle) {
    delete handle;
}

CtwHandle* ctw_clone(const CtwHandle* handle) {
    try {
        if (handle == nullptr) {
            g_last_error = "ctw_clone received null handle";
            return nullptr;
        }
        auto* copy_handle = new CtwHandle(handle->estimator.max_depth);
        copy_handle->estimator.root = handle->estimator.root;
        copy_handle->estimator.undo_log.clear();
        return copy_handle;
    } catch (const std::exception& ex) {
        g_last_error = ex.what();
        return nullptr;
    }
}

int ctw_predict(
    const CtwHandle* handle,
    const int* tokens,
    int len,
    double* p_correct,
    int* deepest_depth,
    int* deepest_total,
    int* deepest_positive,
    int* deepest_negative
) {
    try {
        if (handle == nullptr) {
            g_last_error = "ctw_predict received null handle";
            return -1;
        }
        if (!validate_context(tokens, len)) {
            g_last_error = "ctw_predict received null tokens with positive length";
            return -1;
        }
        const Node* node = &handle->estimator.root;
        int depth = 0;
        int pos = node->count1;
        int neg = node->count0;
        for (int i = 0; i < len; ++i) {
            auto it = node->children.find(tokens[i]);
            if (it == node->children.end()) {
                break;
            }
            node = it->second.get();
            depth = i + 1;
            pos = node->count1;
            neg = node->count0;
        }

        const double log_pw_before = handle->estimator.root.log_pw;
        const double log_pw_after_0 = handle->estimator.root.hypothetical_log_pw(
            tokens,
            len,
            0,
            handle->estimator.max_depth
        );
        const double log_pw_after_1 = handle->estimator.root.hypothetical_log_pw(
            tokens,
            len,
            1,
            handle->estimator.max_depth
        );
        const double raw0 = std::exp(log_pw_after_0 - log_pw_before);
        const double raw1 = std::exp(log_pw_after_1 - log_pw_before);
        const double denom = raw0 + raw1;
        const double prob = (denom > 0.0) ? (raw1 / denom) : 0.5;

        if (p_correct != nullptr) {
            *p_correct = prob;
        }
        if (deepest_depth != nullptr) {
            *deepest_depth = depth;
        }
        if (deepest_total != nullptr) {
            *deepest_total = pos + neg;
        }
        if (deepest_positive != nullptr) {
            *deepest_positive = pos;
        }
        if (deepest_negative != nullptr) {
            *deepest_negative = neg;
        }
        return 0;
    } catch (const std::exception& ex) {
        g_last_error = ex.what();
        return -1;
    }
}

int ctw_update(CtwHandle* handle, const int* tokens, int len, int target) {
    try {
        if (handle == nullptr) {
            g_last_error = "ctw_update received null handle";
            return -1;
        }
        if (!validate_context(tokens, len)) {
            g_last_error = "ctw_update received null tokens with positive length";
            return -1;
        }
        if (target != 0 && target != 1) {
            g_last_error = "ctw_update target must be binary";
            return -1;
        }
        update_with_undo(handle->estimator, &handle->estimator.root, tokens, len, target, handle->estimator.max_depth);
        return 0;
    } catch (const std::exception& ex) {
        g_last_error = ex.what();
        return -1;
    }
}

int ctw_checkpoint(const CtwHandle* handle, std::size_t* marker) {
    try {
        if (handle == nullptr) {
            g_last_error = "ctw_checkpoint received null handle";
            return -1;
        }
        if (marker == nullptr) {
            g_last_error = "ctw_checkpoint received null marker";
            return -1;
        }
        *marker = handle->estimator.undo_log.size();
        return 0;
    } catch (const std::exception& ex) {
        g_last_error = ex.what();
        return -1;
    }
}

int ctw_rollback(CtwHandle* handle, std::size_t marker) {
    try {
        if (handle == nullptr) {
            g_last_error = "ctw_rollback received null handle";
            return -1;
        }
        if (marker > handle->estimator.undo_log.size()) {
            g_last_error = "ctw_rollback marker out of range";
            return -1;
        }
        rollback_to(handle->estimator, marker);
        return 0;
    } catch (const std::exception& ex) {
        g_last_error = ex.what();
        return -1;
    }
}

}  // extern "C"
