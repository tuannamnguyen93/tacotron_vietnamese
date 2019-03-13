from os import listdir
from os.path import isfile, join
import codecs

import os
from os import listdir
from os.path import isfile, join
paths = [
        'txt',
            'wav']
for path in paths:
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        lenfile = len(file)
        name = 'duyen'
        os.rename(path+'/'+file,path+'/'+ name[:15-lenfile] +file)

origin = 'txt'
txt_path = 'metadata.csv'
txts = [f for f in listdir(origin) if isfile(join(origin, f))]
txts.sort()
file = codecs.open(txt_path, "w", "utf-8")
text2 = ''
for txt in txts:
    path = origin+'/'+txt
    with codecs.open(path, 'r', encoding='utf8') as f:
        text = f.readline()
        text = ' '.join(text.split())+'\n'
    text2 = text2 + txt[:-4] +'|'+text.replace('\n','')+'|'+text
with codecs.open(txt_path,'w',encoding='utf8') as f1:
    f1.write(text2)

