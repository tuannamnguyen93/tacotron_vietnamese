from html.parser import HTMLParser
from html.entities import name2codepoint
import re


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

            attrs = {attr[0]:attr[1] for attr in attrs}
            self.tmp_type = tag
            self.tmp_params = attrs

    def handle_endtag(self, tag):
        # print("End tag  :", tag)
        if tag == 'speak':
            pass
        else:
            self.data.append( [self.tmp_text, self.tmp_type, self.tmp_params] )

            self.tmp_text = None
            self.tmp_type = None
            self.tmp_params = None

    def handle_data(self, data):
        # print("Data     :", data)
        self.tmp_text = re.sub(' +',' ',data.strip())
        if self.tmp_type is None and len(self.tmp_text) > 0:
            self.tmp_type = "text"
            self.data.append( [self.tmp_text, self.tmp_type, self.tmp_params] )

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

def sysnthesizer_normal(text):
    pass

def sysnthesizer_say_as(text, interpret_as='spell-out'):

    elif interpret_as == 'characters':
        pass
    elif interpret_as == 'number':
        pass
    elif interpret_as == 'fraction':
        pass
    elif interpret_as == 'unit':
        pass
    elif interpret_as == 'date':
        pass
    elif interpret_as == 'time':
        pass
    elif interpret_as == 'telephone':
        pass
    elif interpret_as == 'address':

def sysnthesizer_break(text, strength='none', time = '0s'):
    # strength is one of ['none', 'x-weak', 'weak', 'medium', 'strong', 'x-strong']
    pass

def sysnthesizer_emphasis(text, level='strong'):
    # level is one of ['strong', 'moderate', 'reduced']
    pass

def sysnthesizer_prosody(text, rate='medium', pitch='medium', volume='medium'):
    # rate : [x-slow, slow, medium, fast, x-fast]
    # pitch : [x-low, low, medium, high, x-high]
    # volume : [silent, x-soft, soft, medium, loud, x-loud]
    pass

MY_SYNTHESIZER_MAPER = {
        'text' : sysnthesizer_normal,
        'say-as' : sysnthesizer_say_as,
        'break' : sysnthesizer_break,
        'emphasis' : sysnthesizer_emphasis,
        'prosody' : sysnthesizer_prosody
    }

if __name__ == "__main__":
    parser = MySSMLParser()

    ssml_str = '<speak>\
      Here are <say-as interpret-as="characters">SSML</say-as> samples.\
      I can pause <break time="3s"/>.\
      I can play a sound\
      I can speak in cardinals. Your number is <say-as interpret-as="cardinal">10</say-as>.\
      Or I can speak in ordinals. You are <say-as interpret-as="ordinal">10</say-as> in line.\
      Or I can even speak in digits. The digits for ten are <say-as interpret-as="characters">10</say-as>.\
      Finally, I can speak a paragraph with two sentences.\
        Step 1, take a deep breath. <break time="200ms"/>\
      Step 2, exhale.\
      Step 3, take a deep breath again. <break strength="weak"/>\
      Step 4, exhale.\
      <prosody rate="slow" pitch="-2st">Can you hear me now?</prosody>\
      <emphasis level="moderate">This is an important announcement</emphasis>\
       <say-as interpret-as="date" format="yyyymmdd" detail="1">\
        1960-09-10\
      </say-as>\
    </speak>'

    ssml_str = ssml_str.lower()
    parser.feed(ssml_str)
    results = parser.get_data()

    ### print out result:
    for res in results:
        print("text value: ", res[0])
        print("sysnthesis type: ", res[1])
        print("sysnthesis params: ", res[2])
        print("systhesis function to call: ", MY_SYNTHESIZER_MAPER[res[1]].__name__)
        print('\n')
    
    # reset parser
    parser.reset_parser()




    print ("Exit")
