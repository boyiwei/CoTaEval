import re

consecutive_newlines = re.compile(r'[\r\n]+')
consecutive_spaces = re.compile(r'[^\S\n]+')
leading_spaces = re.compile(r'\n\s')

def proc_code(text):
    # replace consecutive carriage returns or newlines with a single newline
    text = consecutive_newlines.sub('\n', text)

    # replace consecutive whitespace with a single space (this is a 'squeeze')
    text = consecutive_spaces.sub(' ', text)

    # whitespace has been squashed already, but now strip leading whitespace from each line
    text = leading_spaces.sub('\n', text)

    # so in effect: normalize newlines, remove repeated newlines,
    # compress repeated spaces, and then place everything on the same indentation level

    return text.strip()
