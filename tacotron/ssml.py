import re
from html.entities import name2codepoint
from html.parser import HTMLParser

class MySSMLParser(HTMLParser):
    def __init__(self):
        super(MySSMLParser, self).__init__()
        self.data = []
        self.tmp_text = None
        self.tmp_type = None
        self.tmp_params = None

    def handle_starttag(self, tag, attrs):
        # print("Start tag:", tag)
        if tag == 'speak':
            pass
        else:
            # for attr in attrs:
            # print("     attr:", attr)

            attrs = {attr[0]: attr[1] for attr in attrs}
            self.tmp_type = tag
            self.tmp_params = attrs

    def handle_endtag(self, tag):
        # print("End tag  :", tag)
        if tag == 'speak':
            pass
        else:
            self.data.append([self.tmp_text, self.tmp_type, self.tmp_params])

            self.tmp_text = None
            self.tmp_type = None
            self.tmp_params = None

    def handle_data(self, data):
        # print("Data     :", data)
        self.tmp_text = re.sub(' +', ' ', data.strip())
        if self.tmp_type is None and len(self.tmp_text) > 0:
            self.tmp_type = "text"
            self.data.append([self.tmp_text, self.tmp_type, self.tmp_params])

            self.tmp_text = None
            self.tmp_type = None
            self.tmp_params = None

    def handle_comment(self, data):
        print("Comment  :", data)

    def handle_entityref(self, name):
        c = chr(name2codepoint[name])
        # print("Named ent:", c)

    def handle_charref(self, name):
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        # print("Num ent  :", c)

    def handle_decl(self, data):
        print("Decl     :", data)

    def get_data(self):
        return self.data

    def reset_parser(self):
        self.data = []
        self.tmp_text = None
        self.tmp_type = None
        self.close()
