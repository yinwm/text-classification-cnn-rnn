# coding: utf-8

from __future__ import print_function

import os
import sys
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.train_text_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str



class CnnModel:
    def __init__(self, user_id):
        base_dir = f'data/{user_id}'
        vocab_dir = os.path.join(base_dir, 'vocab.txt')

        save_dir = f'checkpoints/textcnn/{user_id}'
        save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category(user_id)
        self.config.num_classes = len(self.categories)
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]

def predict_user(user_id, text_array):

    cnn_model = CnnModel(user_id)
    for i in text_array:
        print(cnn_model.predict(i))

if __name__ == '__main__':

    if len(sys.argv) != 2 :
        raise ValueError("""usage: python predict_cnn.py [$USER_ID]""")

    user_id = sys.argv[1]

    test_demo = ['胸部太小 如何搭配上衣(图)导读：胸小女生你们还在为别人叫你太平公主发愁吗？担心这个秋冬的厚外套更加让自己没有身材了？还在为你的胸部太小不会搭配衣服伤心吗？看完这篇文章就是你摆摊胸小的时候了，快乐学习胸部太小的女生如何搭配衣服吧！点评：此种搭配只限黑白色，而且顺序不能颠倒，一定要是内搭白色，外衫黑色才能达到胸部突出的效果。在外衫的选择上镂空不要太大，套头衫效果最好。搭配腰带可以起到收紧修身的效果，让突出的胸部更加明显。点评：大U领一直开到胸部，露出内搭的黑色抹胸，胸围处另用伸缩的竖条织法起到承托胸部的作用，使胸部看起来更坚挺有型。MSVIVI这款采用精细的16 针针织法，定型效果更好。灰白配色的层次感带来丰富的视觉效果，还能感觉到胸部的曲线，很诱人哦。从视觉上看起来特别丰满，定能弥补平日的遗憾！点评：胸部中间用抓皱的方法做了一条明显的分割线，将整个胸部从整体视线中提升出来，突出的皱褶能让你从B杯罩提成到C的视觉效果，外面挑一件小西装使胸部若隐若现，更加吸引眼球。InWear来自丹麦的女装品牌，简洁的女性化风格深受白领女性的喜爱。点评：飞机场妹妹不妨选这款胸部带蝴蝶结的衣服，重叠的皱褶增加胸部饱满感，点缀的蝴蝶结可以掩饰胸部的不足，在走动之间还能跌宕起伏给人以视觉上的假象。大领子的设计，也容易把人的注意力转移到你的香肩及以上的部位。点评：两件套叠加的层次感，恰到好处的起到了，视觉上的承托胸部的作用，内层垂坠贴身，露出胸部优美曲线，领子开口大小恰到好处的露出性感锁骨，外层的叠加突出胸部的存在感。两件套完美搭配，呈现大胸部的你。',
                 '新版莎翁《暴风雨》网罗奥斯卡影帝影后(图)早报记者 张悦编译 《狮子王()》音乐剧及电影《弗里达》的导演朱莉·泰摩，最近将目光聚焦莎翁名作，计划把《暴风雨》再次搬上大银幕。演员方面更是请来超豪华的“奥斯卡级”全明星阵容，其中包括杰雷米·艾恩斯和去年的奥斯卡影后海伦·米伦。两人曾在电视版的《伊丽莎白女王》中有过合作。 《暴风雨》是莎士比亚最后一部作品，也被称为“传奇剧”，讲的是一个童话故事，描述普洛斯佩罗公爵因为钻研魔法而被弟弟和那不勒斯国王陷害，沦落到荒岛上与女儿米兰达相依为命。一场暴风雨把剧中的人物与世隔绝，把他们带到一个神秘的小岛上。剧名《暴风雨》并不仅仅影射自然界的暴风雨，它尤其要表现剧中人物那充满汹涌纷繁情感的内心世界。海伦·米伦将饰演这个魔法小岛上的女王。普洛斯佩罗公爵的女儿米兰达则由英国新人女演员费莉西蒂·琼斯扮演。曾出演《血钻》的迪蒙·亨素和《香水》的男主角本·惠肖将扮演岛上的居民———分别为被魔法变了形的奴隶卡利班和会飞的精灵埃里尔。杰雷米·艾恩斯将饰演那不勒斯国王阿隆索，英国喜剧明星拉塞尔·布兰德将饰演一名小丑，“章鱼博士”阿尔弗雷德·莫里纳饰演酗酒成性的管家斯蒂法诺。老牌影星杰弗里·拉什也正在商谈加盟该片，有望扮演普洛斯佩罗公爵的参谋冈扎洛。其中，海伦·米伦是去年的奥斯卡影后，杰雷米·艾恩斯曾是1991年的奥斯卡影帝(《命运的逆转()》)，杰弗里·拉什也曾凭借《闪亮的风采》获得奥斯卡最佳男主角奖。黑人影星迪蒙·亨素则曾凭借《血钻》和《在美国》两次获得奥斯卡最佳男配角提名。 该片计划于11月在夏威夷开拍，米拉麦克斯将负责该片在所有英语国家的发行。执导音乐剧出身的导演朱莉·泰摩，曾执导过奥斯卡提名影片《弗里达》和《狮子王》的音乐剧。去年，她还将一系列“披头士”乐队的歌曲融入一部青春爱情电影，执导了音乐片《穿越宇宙》。泰摩对改编莎剧也不陌生。此次改编《暴风雨》之前，她还曾把莎剧《泰特斯·安特洛尼克斯》改编成颇具舞台感的电影《泰特斯》(1999年)，该片由安东尼·霍普金斯、杰西卡·兰格、乔纳森·里斯-迈耶斯等主演。 ',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00',
                 '某个新技术性能的大幅迅速提高，有心的观察者慢慢可以看到新 的产业价值链的形成，取代老技术的价值链。 新老价值链的区别， 不仅仅在于简单的几个性能上的可比的参数，而是当新技术在部分性 能超越老的模型之后，形成一个完全不同的生态下的新思维模型，就 像电脑不只是文字处理还可以互联共享数据，iphone 不只是用手机上 网发 email 还可以支持各种复杂的移动应用, 等等']

    predict_user(user_id, test_demo)


