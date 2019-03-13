from . import cmudict

_pad        = '_'
_eos        = '~'
_characters = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ!\'(),-.:;? '
# _characters = 'AÁÀẠẢÃĂẰẮẶẲẴÂẤẦẬẨẪBCDĐEÉÈẺẼẸÊẾỀỂỄỆFGHIÍÌỊỈĨJKLMNOÓÒỌỎÕƠỚỜỢỠỞÔỐỒỘỖỔPQRSTUÚÙỤỦŨƯỨỪỰỮỬVWXYZaáàạảãăắằặẳẵâấầậẩẫbcdđeéèẻẽẹêếềểễệfghiíìịỉĩjklmnoóòọỏõơớờợỡởôốồộỗổpqrstuúùụủũưứừựữửvwxyz!\'(),-.:;? '
#_characters= 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ!\'(),-.:;? '
#_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + _arpabet