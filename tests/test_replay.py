"""Replay / stdout / fuel-log parsing (bctools.replay)."""

import struct

from bctools import replay


def test_parse_stdout_strips_nul_padding_and_tags():
    text = '\x00\x00T {"r":1,"c":"Varg","f0":100,"f1":40}\n\x00E {"r":2,"k":"dead"}\nX {"ctx":"move","err":"e"}\nnoise\n'
    log = replay.parse_stdout(text)
    assert log.turns[0]["c"] == "Varg" and log.events[0]["k"] == "dead" and log.errors[0]["ctx"] == "move"
    assert log.other == ["noise"]


def test_parse_fuel_log():
    body = struct.pack("<3Q", 155623, 100000, 42000)
    data = replay.FUEL_MAGIC + bytes(5) + struct.pack("<8Q", *([100000] * 8)) + body
    budgets, fuel = replay.parse_fuel(data)
    assert fuel == [155623, 100000, 42000] and len(budgets) == 8
    log = replay.TeamLog(fuel=fuel)
    assert replay.exhausted_turns(log) == 1
