#pragma once
#include <cstdint>

// Event types for the backtest simulation clock.
enum class EventType : uint8_t {
    GAME_TICK,   // periodic state update during a live game
    GAME_FINAL,  // game over; combo_won is set in data
};

// POD payload attached to every event.  Floats first to avoid padding.
struct GameTick {
    float tau_min;           // minutes remaining (0 at game end)
    float score_diff;        // home_score - away_score
    float fair_value;        // model P(combo wins | state)
    float executable_bid;    // current best exit bid
    float entry_price;       // cost basis per contract
    uint8_t legs_live;
    uint8_t legs_completed;
    uint8_t legs_total;
    uint8_t combo_won;       // valid only for GAME_FINAL (0 or 1)
    uint8_t _pad[4];
};

struct Event {
    int64_t  timestamp_ms;   // simulation clock (ordering key)
    uint32_t game_id;
    EventType type;
    uint8_t  _pad[3];
    GameTick data;
};
