"""군사 단위 변환 (길이·속도·무게·온도·각도(밀)·압력 등)."""

from __future__ import annotations

import math

# 기준 단위로 변환하는 계수 (길이: m, 속도: m/s, 무게: kg, 부피: L, 각도: deg, 압력: kPa)
_LENGTH = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "nm": 1852.0, "nmi": 1852.0, "yd": 0.9144, "ft": 0.3048, "in": 0.0254}
_SPEED = {"m/s": 1.0, "km/h": 1000.0 / 3600.0, "kph": 1000.0 / 3600.0, "mph": 1609.344 / 3600.0, "kn": 1852.0 / 3600.0, "kt": 1852.0 / 3600.0, "kts": 1852.0 / 3600.0, "ft/s": 0.3048}
_MASS = {"kg": 1.0, "g": 0.001, "t": 1000.0, "lb": 0.45359237, "oz": 0.028349523125}
_VOLUME = {"l": 1.0, "ml": 0.001, "gal": 3.785411784, "qt": 0.946352946}
_ANGLE = {"deg": 1.0, "mil": 360.0 / 6400.0, "mils": 360.0 / 6400.0, "rad": 57.29577951308232, "grad": 0.9}
_PRESSURE = {"kpa": 1.0, "hpa": 0.1, "mbar": 0.1, "bar": 100.0, "psi": 6.894757293168, "atm": 101.325, "mmhg": 0.133322387415, "inhg": 3.38638866667}

_ALIASES = {
    "미터": "m", "킬로미터": "km", "마일": "mi", "해리": "nm", "야드": "yd", "피트": "ft", "인치": "in",
    "노트": "kn", "킬로그램": "kg", "파운드": "lb", "리터": "l", "갤런": "gal",
    "도": "deg", "밀": "mil", "라디안": "rad", "섭씨": "c", "화씨": "f", "켈빈": "k",
    "℃": "c", "°c": "c", "℉": "f", "°f": "f", "°": "deg",
}

_TABLES = [("길이", _LENGTH), ("속도", _SPEED), ("무게", _MASS), ("부피", _VOLUME), ("각도", _ANGLE), ("압력", _PRESSURE)]


def _norm(unit: str) -> str:
    u = unit.strip().lower()
    return _ALIASES.get(u, u)


def _temperature(value: float, src: str, dst: str) -> float:
    to_c = {"c": lambda v: v, "f": lambda v: (v - 32) * 5 / 9, "k": lambda v: v - 273.15}
    from_c = {"c": lambda v: v, "f": lambda v: v * 9 / 5 + 32, "k": lambda v: v + 273.15}
    return from_c[dst](to_c[src](value))


def _fmt_result(x: float, sig: int = 4) -> str:
    """유효숫자 약 4자리로, 지수 표기 없이 천 단위 구분 기호를 넣어 표시한다."""
    if x == 0 or not math.isfinite(x):
        return f"{x:g}"
    magnitude = math.floor(math.log10(abs(x)))
    decimals = min(max(0, sig - 1 - magnitude), 12)
    s = f"{x:,.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in {"-0", ""} else s


def _fmt_input(x: float) -> str:
    """입력값은 자릿수를 잃지 않도록 그대로(지수 표기 없이) 표시한다."""
    x = float(x)
    if not math.isfinite(x):
        return f"{x:g}"
    if x == int(x) and abs(x) < 1e15:
        return f"{int(x):,}"
    s = f"{x:,.12f}".rstrip("0").rstrip(".")
    return s if s not in {"0", "-0"} else f"{x:g}"


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """단위를 변환한다. 같은 범주(길이/속도/무게/부피/각도/압력/온도)끼리만 가능하다."""
    src, dst = _norm(from_unit), _norm(to_unit)
    if src in {"c", "f", "k"} and dst in {"c", "f", "k"}:
        out = _temperature(float(value), src, dst)
        return f"{_fmt_input(value)} {from_unit} = {out:.2f} {to_unit} (온도)"
    for name, table in _TABLES:
        if src in table and dst in table:
            out = float(value) * table[src] / table[dst]
            return f"{_fmt_input(value)} {from_unit} = {_fmt_result(out)} {to_unit} ({name})"
    raise ValueError(
        f"변환할 수 없는 단위 조합: {from_unit!r} → {to_unit!r}. "
        "지원 단위: 길이(m, km, mi, nm, yd, ft, in), 속도(m/s, km/h, mph, kn), 무게(kg, g, t, lb, oz), "
        "부피(l, ml, gal), 각도(deg, mil, rad), 압력(kPa, hPa, bar, psi), 온도(C, F, K)"
    )
