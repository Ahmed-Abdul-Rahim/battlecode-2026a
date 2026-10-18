"""Python side of the Rust parity fixtures, codegen drift and replay parsing."""

import json
import struct
from pathlib import Path

import numpy as np

from bctools import replay
from ml.distill import quant
from ml.features import gen, schema_gen
from ml.uda.dann import invariant_mask, lambda_schedule
from ml.util import crc16, q

ROOT = Path(__file__).resolve().parents[1]


def test_crc_and_quantiser_match_rust():
    assert crc16(b"123456789") == 0x29B1
    assert q(0.5) == 64 and q(-0.5) == -64 and q(9) == 127 and q(-9) == -127


def test_int8_forward_matches_fixture():
    fixture = json.loads((ROOT / "tests" / "fixtures" / "int8_reference.json").read_text())
    schema_hash, layers = quant.decode(bytes.fromhex(fixture["blob_hex"]))
    assert schema_hash == 0xBDF9
    for case in fixture["cases"]:
        assert (
            quant.forward(layers, case["input"], [tuple(p) for p in case["film"]]).tolist() == case["logits"]
        )


def test_blob_round_trip():
    layers = [quant.Layer(np.array([[1, -2], [3, 4]], np.int8), np.array([5, -6], np.int32), 2, True, 0)]
    _, back = quant.decode(quant.encode(0x1234, layers))
    assert back[0].weights.tolist() == [[1, -2], [3, 4]] and back[0].film_offset == 0


def test_generated_schema_is_current():
    assert gen.main(["--check"]) == 0
    s = gen.load()
    assert s["hash"] == schema_gen.SCHEMA_HASH
    assert len(schema_gen.FEATURE_NAMES) == schema_gen.N_FEATURES == len(invariant_mask())


def test_dann_lambda_ramp():
    assert lambda_schedule(0.0) == 0.0
    assert 0.99 < lambda_schedule(1.0) <= 1.0


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
