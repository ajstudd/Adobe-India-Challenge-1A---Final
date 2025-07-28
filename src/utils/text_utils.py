
def is_valid_heading(text):
    """
    Simple heuristic to check if text is likely a heading.
    """
    return len(text) < 100 and not text.endswith(".")

def clean_text(text):
    """
    Cleans and normalizes text.
    """
    return text.strip()

def is_uppercase(text):
    return text.isupper()

def is_title_case(text):
    return text.istitle()

def contains_mojibake(text):
    """
    Heuristic to check if text has mojibake (garbled characters).
    """
    # Common mojibake byte ranges and patterns
    suspicious_chars = [
        'ã', 'Â', '�', '¼', '½', '¾', 'ÿ', 'ð', 'þ', '¢', '¤', '¦', '§', '¨', '©', '«', '¬', '®', '¯',
        '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿', '×', '÷',
        '\ufffd', # replacement char
    ]
    # Also check for long runs of non-ASCII or replacement chars
    if any(char in text for char in suspicious_chars):
        return True
    if text.count('\ufffd') > 0:
        return True
    # Heuristic: if >30% of chars are non-ASCII, likely mojibake
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) > 0.3:
        return True
    return False
