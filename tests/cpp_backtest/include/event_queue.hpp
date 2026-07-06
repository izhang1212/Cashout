#pragma once
#include "event.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

// Time-sorted event queue for batch backtesting.
//
// Design: accumulate events via push(), call finalize() once (sorts
// ascending by timestamp_ms in O(n log n)), then drain with next().
// This is faster than a priority_queue (heap) for backtesting because
// events are generated in bulk and consumed front-to-back — sequential
// access is cache-friendly vs. random heap traversal.
//
// For real-time use, replace finalize()+next() with a std::priority_queue.
class EventQueue {
public:
    explicit EventQueue(size_t reserve_cap = 65536) {
        _events.reserve(reserve_cap);
    }

    void push(Event e) { _events.push_back(std::move(e)); }

    // Must be called once after all events are pushed, before any next().
    void finalize() {
        std::sort(_events.begin(), _events.end(),
                  [](const Event& a, const Event& b) {
                      return a.timestamp_ms < b.timestamp_ms;
                  });
        _pos = 0;
    }

    bool   empty()           const { return _pos >= _events.size(); }
    size_t size()            const { return _events.size(); }
    size_t events_processed() const { return _pos; }

    const Event& peek() const { return _events[_pos]; }

    Event next() { return _events[_pos++]; }

    void clear() { _events.clear(); _pos = 0; }

private:
    std::vector<Event> _events;
    size_t _pos = 0;
};
