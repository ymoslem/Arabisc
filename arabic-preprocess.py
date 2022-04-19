import re
import string
import sys


source = sys.argv[1]
output = source+".clean"


# Remove diacritics (Tashkil)
def remove_diacritics(text):
    arabic_diacritics = re.compile("""
                                \u064E | #Fatha
                                \u064B | #Tanwin Fath
                                \u0650 | #Kasra
                                \u064D | #Tanwin Kasr
                                \u064F | #Damma
                                \u064C | #Tanwin Damm
                                \u0652 | #Sukun
                                \u0651 | #Shadda
                                \u0640 | #Tatwil/Kashida
                             """, re.VERBOSE)

    text = re.sub(arabic_diacritics, '', text)

    return text


# Remove English Characters
def remove_latin(text):
    english_characters = re.compile(r'[a-zA-Z]')
    text = re.sub(english_characters, '', text)

    return text


# Remove the rest of punctuation marks
def remove_punctuation(text):
    arabic_punctuations =  '''`÷×؛<>_()*&^%][ـ،:"؟.,'{}~¦+|!”…“–ـ/$£•●'''
    english_punctuations = string.punctuation
    numbers = "1234567890١٢٣٤٥٦٧٨٩٠"
    bad_characters = "�¿áóóó□"
    punctuations_list = arabic_punctuations + english_punctuations + numbers +  bad_characters

    replace_slash = str.maketrans('/', ' ', '')
    text = text.translate(replace_slash)
    remove_punc = str.maketrans('', '', punctuations_list)
    text = text.translate(remove_punc)

    return text


# Start processing the input file
with open(source) as f:
    text = f.read()

# Split on punctuation and remove duplicates
text = re.split(r'\. |\.\n|\!\n|\؟\n|\n', text)
text = list(set(text))
arabic_characters = "اأإبتثجحخدذرزسشصضطظعغفقكلمنهويىة"

with open(output, "w+") as clean:
    for segment in text:
        segment = segment.strip()
        segment = remove_diacritics(segment)
        segment = remove_punctuation(segment)
        segment = remove_latin(segment)
        segment = segment.strip()
        segment = " ".join(segment.split())                   # remove extra white-spaces
        segment = " ".join(segment.split()[:15])              # trancate to 15 tokens
        if segment != "" and len(segment.split())>3:          # not empty and > 3 tokens (one is <s>)
            segment = "<s> " + segment                        # adding a start token
            if segment[4] in arabic_characters:
                clean.write(segment + "\n")


print("Done")
