import re
from text import cleaners
from text.symbols import symbols
from text.cleaner_vietnamese import cleaner
# from underthesea import word_tokenize as word_sent
#
# __specChar__ = {u'&': u'và', u'@': u'a còng', u'^': u'mũ', u'$': u'đô la', u'%': u'phần trăm', u'*': u'sao',
#                 u'+': u'cộng', u'>': u'dấu lớn', u'<': u'dấu bé', u'/': u'phần', u'=': u'bằng'}
# __number__ = {0: u'không', 1: u'một', 2: u'hai', 3: u'ba', 4: u'bốn', 5: u'năm', 6: u'sáu', 7: u'bảy', 8: u'tám',
#               9: u'chín', 10: u'mười'}
# __currency__ = {u'VND': u'việt nam đồng', u'USD': 'đô la mỹ'}
# __doluong__ = {u'km': u'ki lô mét', u'cm': u'xen ti mét', u'dm': u'đề xi mét', u'mm': u'mi li mét',
#                u'nm': u'na nô mét'}
# __cannang__ = {u'kg': u'ki lô gam', u'g': 'gờ ram'}
#
# specChar = u'@^$%*+-><='


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(texts, cleaner_names):

  text = texts
  text = cleaner(text).do()

  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  # Append EOS token
  sequence.append(_symbol_to_id['~'])
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

# end
