from parlay_follower.market_data.orderbook import (BookLevel, best_bid,
                                                   executable_proceeds,
                                                   parse_yes_bids)


def test_parse_and_best_bid():
    raw = {"yes": [[40, 100], [45, 50], [42, 30]]}
    levels = parse_yes_bids(raw)
    assert best_bid(levels) == 0.45
    assert [l.price for l in levels] == [0.45, 0.42, 0.40]


def test_executable_proceeds_walks_the_book():
    levels = [BookLevel(0.45, 50), BookLevel(0.42, 30), BookLevel(0.40, 100)]
    proceeds, avg = executable_proceeds(levels, 60)
    assert abs(proceeds - (50 * 0.45 + 10 * 0.42)) < 1e-9
    assert avg < 0.45  # slippage below top-of-book


def test_thin_book_is_pessimistic():
    levels = [BookLevel(0.45, 10)]
    proceeds, avg = executable_proceeds(levels, 100)
    assert abs(proceeds - 4.5) < 1e-9
    assert avg == 4.5 / 100  # unfilled remainder valued at 0
