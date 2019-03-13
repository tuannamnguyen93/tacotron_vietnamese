# Ky tu
_char_ = {u'a':'a', u'b':'bê', u'c':'xê', u'd':'dê', u'e':'e', u'f':'ép', u'g':'gờ', u'h':'hát', u'i':'i', u'j':'di',
          u'k': 'ca', u'l':'lờ', u'm':'mờ', u'n':'nờ', u'o':'ô', u'p':'pê', u'q':'quy', u'r':'e rờ', u's':'sờ', u't':'tê',
          u'u': 'iu', u'v':'vê', u'w':'vê kép', u'x':'ích', u'y':'i', u'z':'dét'}

# Ky tu dac biet
_spec_char = {u'&': u' và ', u'@': u'a còng', u'^': u'mũ', u'$': u'đô la',
              u'%': u'phần trăm', u'*': u'sao', u'+': u'cộng', u'>': u'lớn hơn',
              u'<': u'bé hơn', u'=': u'bằng', u'/': u'trên', 
              u'(': u'mở ngoặc',  u')': u'đóng ngoặc',
              }

# Chu so
_number = {0: u'không', 1: u'một', 2: u'hai', 3: u'ba', 4: u'bốn', 5: u'năm', 6: u'sáu', 7: u'bảy', 8: u'tám',
           9: u'chín', 10: u'mười'}

# Don vi tien te
_currency = {u'vnd': u'việt nam đồng', u'usd': 'đô la mỹ', u'eur': u'ơ rô'}

# Don vi do luong
_d_unit = {u'km': u'ki lô mét', u'cm': u'xen ti mét', u'dm': u'đề xi mét', 
			u'mm': u'mi li mét', u'nm': u'na nô mét', u'l':u'lít',
           u'm2': u'mét vuông', u'm3': u'mét khối',
           u'hz': u'héc', u'm': u'mét',
           u'h': u'giờ', u'p': u'phút', u's': u'giây'
           }

# Don vi can nang
_w_unit = {u'kg': u'ki lô gam', u'g': 'gam'}


# Tu dien viet tat
def short_dict():
    d = {}
    with open("dictionary/short_dict.txt") as f:
        for line in f:
            (key, val) = line.split(",")
            d[str(key).lower()] = str(val).lower()
    return d


def name_dict():
    d = {}
    with open("dictionary/name_dict.txt") as f:
        for line in f:
            (key, val) = line.split(",")
            d[str(key).lower()] = str(val).lower()
    return d


def location_dict():
    d = {}
    with open("dictionary/location_dict.txt") as f:
        for line in f:
            (key, val) = line.split(",")
            d[str(key).lower()] = str(val).lower()
    return d


def name_brand_dict():
    d = {}
    with open("dictionary/name_brand_dict.txt") as f:
        for line in f:
            (key, val) = line.split(",")
            str(key).replace(" ", "")
            d[str(key).lower()] = str(val).lower()
    return d


def capital_dict():
    d = {}
    with open("dictionary/capital_dict.txt") as f:
        for line in f:
            (key, val) = line.split(",")
            d[str(key).lower()] = str(val).lower()
    return d


def voca_dict():
    d = {}
    with open("dictionary/capital_dict.txt") as f:
        for line in f:
            (key, val) = line.split(",")
            d[str(key).lower()] = str(val).lower()
    return d


def countries_dict():
    d = {}
    with open("dictionary/countries_dict.txt") as f:
        for line in f:
            (key, val) = line.split(",")
            d[str(key).lower()] = str(val).lower()
    return d

