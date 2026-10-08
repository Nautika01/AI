"""군 일시군(DTG) 변환 도구."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_DTG_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})([A-IK-Z])\s*([A-Z]{3})\s*(\d{2}|\d{4})$")


def zone_offset(letter: str) -> timedelta:
    """군 시간대 문자 → UTC 오프셋. J(현지 시간)는 지원하지 않는다."""
    letter = letter.strip().upper()
    if letter in {"J", "LOCAL", "현지", "현지시간", "현지 시간"}:
        raise ValueError(f"지원하지 않는 시간대: {letter!r} (J·현지 시간은 위치에 따라 달라 변환할 수 없습니다. Z, I 등 시간대 문자나 KST/UTC 를 지정하십시오)")
    if len(letter) != 1:
        raise ValueError(f"시간대는 한 글자 군 시간대 문자(Z, I 등) 또는 KST/UTC 여야 합니다: {letter!r}")
    if letter == "Z":
        return timedelta(0)
    if "A" <= letter <= "I":
        return timedelta(hours=ord(letter) - ord("A") + 1)
    if "K" <= letter <= "M":
        return timedelta(hours=ord(letter) - ord("K") + 10)
    if "N" <= letter <= "Y":
        return timedelta(hours=-(ord(letter) - ord("N") + 1))
    raise ValueError(f"지원하지 않는 시간대 문자: {letter!r} (J는 현지 시간을 뜻하므로 변환 불가)")


def zone_letter(offset_hours: int) -> str:
    if offset_hours == 0:
        return "Z"
    if 1 <= offset_hours <= 9:
        return chr(ord("A") + offset_hours - 1)
    if 10 <= offset_hours <= 12:
        return chr(ord("K") + offset_hours - 10)
    if -12 <= offset_hours <= -1:
        return chr(ord("N") - offset_hours - 1)
    raise ValueError(f"시간대 문자로 표현할 수 없는 오프셋: {offset_hours}")


def parse_dtg(value: str) -> datetime:
    """`071430IOCT26` 형식의 일시군을 timezone-aware datetime으로 변환한다."""
    m = _DTG_RE.match(value.strip().upper().replace(" ", ""))
    if not m:
        raise ValueError(f"일시군 형식이 아닙니다: {value!r} (예: 071430IOCT26)")
    day, hour, minute, zone, mon, year = m.groups()
    if mon not in _MONTHS:
        raise ValueError(f"알 수 없는 월 표기: {mon}")
    y = int(year) if len(year) == 4 else 2000 + int(year)
    tz = timezone(zone_offset(zone))
    return datetime(y, _MONTHS.index(mon) + 1, int(day), int(hour), int(minute), tzinfo=tz)


def format_dtg(dt: datetime, zone: str = "Z") -> str:
    zone = zone.upper()
    target = dt.astimezone(timezone(zone_offset(zone)))
    return f"{target.day:02d}{target.hour:02d}{target.minute:02d}{zone}{_MONTHS[target.month - 1]}{target.year % 100:02d}"


def _parse_any(value: str) -> datetime:
    value = value.strip()
    try:
        return parse_dtg(value)
    except ValueError:
        pass
    iso = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as e:
        raise ValueError(f"일시군(예: 071430IOCT26) 또는 ISO 8601(예: 2026-10-07T14:30+09:00) 형식이어야 합니다: {value!r}") from e
    if dt.tzinfo is None:
        raise ValueError("ISO 형식에는 시간대 오프셋이 필요합니다 (예: 2026-10-07T14:30+09:00)")
    return dt


def convert_dtg(value: str, to_zone: str = "Z") -> str:
    """일시군 또는 ISO 시각을 다른 시간대의 일시군으로 변환해 설명 문자열을 돌려준다."""
    dt = _parse_any(value)
    to_zone = to_zone.strip().upper()
    if to_zone in {"KST", "I"}:
        to_zone = "I"
    elif to_zone in {"UTC", "Z", "GMT"}:
        to_zone = "Z"
    result = format_dtg(dt, to_zone)
    target = dt.astimezone(timezone(zone_offset(to_zone)))
    return (
        f"입력: {value.strip()} → {result}\n"
        f"ISO 8601: {target.isoformat(timespec='minutes')}\n"
        f"(시간대 {to_zone} = UTC{_fmt_offset(zone_offset(to_zone))})"
    )


def current_dtg(zone: str = "I", now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    zone = zone.strip().upper()
    zone = {"KST": "I", "UTC": "Z", "GMT": "Z"}.get(zone, zone)
    return f"{format_dtg(now, zone)} ({now.astimezone(timezone(zone_offset(zone))).isoformat(timespec='minutes')})"


def _fmt_offset(td: timedelta) -> str:
    hours = int(td.total_seconds() // 3600)
    return f"{hours:+d}"
