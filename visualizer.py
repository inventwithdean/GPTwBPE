from tokenizer import Tokenizer
from wordcloud import WordCloud, STOPWORDS    
import matplotlib.pyplot as plt


tokenizer = Tokenizer()

a = []
for i in tokenizer.merges.values():
    a.append(tokenizer.bytes_dict[i].decode(encoding='utf-8', errors="replace"))

string = " ".join(a)

cloud = WordCloud(width=400, height=400, background_color="white", min_font_size=10, stopwords=set(STOPWORDS)).generate(string)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_axis_off()
fig.tight_layout(pad=0)
ax.imshow(cloud)
fig.savefig("./out.png")