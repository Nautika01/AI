"""NATO 음성 문자(포네틱 알파벳) 변환."""

from __future__ import annotations

NATO = {
    "A": "Alfa", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo", "F": "Foxtrot",
    "G": "Golf", "H": "Hotel", "I": "India", "J": "Juliett", "K": "Kilo", "L": "Lima",
    "M": "Mike", "N": "November", "O": "Oscar", "P": "Papa", "Q": "Quebec", "R": "Romeo",
    "S": "Sierra", "T": "Tango", "U": "Uniform", "V": "Victor", "W": "Whiskey", "X": "X-ray",
    "Y": "Yankee", "Z": "Zulu",
}
DIGITS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Tree", "4": "Fower", "5": "Fife",
    "6": "Six", "7": "Seven", "8": "Ait", "9": "Niner",
}
SYMBOLS = {"-": "Dash", ".": "Decimal", "/": "Slant", " ": "(띄움)"}


def phonetic_spell(text: str) -> str:
    """문자열을 NATO 음성 문자로 한 글자씩 풀어 쓴다. 한글 등 지원하지 않는 문자는 그대로 둔다."""
    if not text.strip():
        raise ValueError("변환할 문자열이 비어 있습니다.")
    words: list[str] = []
    for ch in text.strip():
        up = ch.upper()
        if up in NATO:
            words.append(NATO[up])
        elif ch in DIGITS:
            words.append(DIGITS[ch])
        elif ch in SYMBOLS:
            words.append(SYMBOLS[ch])
        else:
            words.append(ch)
    return f"{text.strip()} → " + " ".join(words)
