import matplotlib.pyplot as plt     #数学绘图库
import jieba            #分词库
from wordcloud import WordCloud,ImageColorGenerator   #词云库
from scipy.misc import imread
text = open(r'.\1.txt','r').read()
#print(text)

#分开词
cut_text = jieba.cut(text)
result = ' '.join(cut_text)
print(result)

#导入图片
image =imread('.\1,jpg')

#生成云图
wc = WordCloud(
    #字体路径
    font_path=r'.\simsun.ttc',
    #背景颜色
    background_color='white',
    max_font_size=50,
    min_font_size=10,
    #词云形状
    mask=image
)

wc.generate(result)

#从背景里提取背景颜色
image_color = ImageColorGenerator(image)
wc.recolor(color_func=image_color)
wc.to_file(r'.\wordcloud.png')

#显示图片
#图片名称
plt.figure('wordcould')
plt.imshow(wc)
plt.axis('off')
plt.show()