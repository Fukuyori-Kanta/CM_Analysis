import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import time
import csv
import os
from logging import getLogger, StreamHandler, DEBUG

# プロキシ設定
#os.environ["https_proxy"] = "http://wwwproxy.kanazawa-it.ac.jp:8080"

# モジュールのキャッシュ設定
os.environ ["TFHUB_CACHE_DIR"] = r'C:\Users\fukuyori\AppData\Local\Temp\tfhub_modules'

# ログ設定
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

# パス設定
input_path = r"C:\Users\MatsuiLab\Desktop\Fukuyori\Movie"   # 入力先パス
csv_input_path = r'C:\Users\MatsuiLab\Desktop\Fukuyori\MovieList_200508.csv'    # CSV入力パス
result_noun_path = r"C:\Users\MatsuiLab\Desktop\Fukuyori\Scene_Count.csv"       # CSV出力パス
cut_input_path = r'C:\Users\MatsuiLab\Desktop\Fukuyori\Cut' # カット画像の保存先パス    # o

# tensorflowのバージョン表示
print(tf.version.VERSION)

# 使用出来るGPUを表示
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())


def run_detector(detector, path):

  # 画像を読み込んで detector に入力できる形式に変換
  img = Image.open(path) # Pillow(PIL)
  if img.mode == 'RGBA' :
    img = img.convert('RGB')
  converted_img = img.copy()
  converted_img = converted_img.resize((227,227),Image.LANCZOS) # 入力サイズに縮小
  converted_img = np.array(converted_img, dtype=np.float32)     # np.arrayに変換
  converted_img = converted_img / 255. # 0.0 ～ 1.0 に正規化
  converted_img = converted_img.reshape([1,227,227,3])
  converted_img = tf.constant(converted_img)

  t1 = time.time()
  result = detector(converted_img) # 一般物体検出（本体）
  t2 = time.time()
  #print(f'検出時間 : {t2-t1:.3f} 秒' )

  # 結果をテキスト出力するための準備
  r = {key:value.numpy() for key,value in result.items()}
  boxes =       r['detection_boxes']
  scores =      r['detection_scores']
  decode = np.frompyfunc( lambda p : p.decode('ascii'), 1, 1)
  class_names = decode( r['detection_class_entities'] )

  label = []
  # スコアが 0.25 以上の結果（n件）についてテキスト出力
  #print(f'検出オブジェクト' )
  n = np.count_nonzero(scores >= 0.25 )
  for i in range(n):
    y1, x1, y2, x2 = tuple(boxes[i])
    x1, x2 = int(x1*img.width), int(x2*img.width)
    y1, y2 = int(y1*img.height),int(y2*img.height)
    t = f'{class_names[i]:10} {100*scores[i]:3.0f}%  '
    t += f'({x1:>4},{y1:>4}) - ({x2:>4},{y2:>4})'
    print(t)
    label.append(class_names[i] + '(' + str(round(100*scores[i], 2)) + '%)')

  return label

def read_csv(file_path):
    """
    CSVファイルを読み込んで、その結果を返す関数

    Parameters
    ----------
    file_path : str
        読み込むCSVファイルのパス

    Returns
    -------
    l : list
        読み込んだ結果を返すリスト
    """
    csvfile = open(file_path, 'r', encoding='utf-8')
    reader = csv.reader(csvfile)
    #header = next(reader)  # ヘッダーを読み飛ばしたい時

    l = []
    for row in reader:
        l.append(row)

    return l

def write_csv(data, cut_img_path):
    """
    データを受け取り、CSVに書き出す関数

    Parameters
    ----------
    data : list
        出力するデータ
    cut_img_path : str
        保存先フォルダのパス
    """
    with open(cut_img_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def exe():
    # 動画IDの取得
    movie_list = read_csv(csv_input_path)   # CSVファイルの情報をリストに格納
    movie_num = len(movie_list) # 動画の個数
    video_id_list = [movie_list[i][1] for i in range(movie_num)]    # 全動画IDのリスト

    # パス
    #cut_input_path = r'C:\Lab2\Cut\D201061228' # カット画像の保存先パス
    print('モジュールの読み込み中...')
    # モジュールの読み込み
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures['default']
    print('モジュール読み込み完了')

    label_result = []   # 最終結果用のラベルリスト
    for video_id in video_id_list:
        print('==========')
        print(video_id)
        print('==========')
        cut_img_path = cut_input_path + '\\' + video_id # 保存先パス

        # ファイルの読み込み
        files = os.listdir(cut_img_path)
        label_list = []     # ラベルのリスト

        for i in range(len(files)):
            path = cut_img_path + '\\cut_' + str(i+1) + '.jpg'  # カットのパス
            cut_no = 'cut_' + str(i+1)  # カット番号
            print('[' + cut_no + ']')

            # 物体検出・認識
            label = run_detector(detector, path)            # 検出結果のラベル
            label_list.append([video_id, cut_no, label])    # [動画ID, カット番号, ラベルリスト]

            print('-----------------')
        label_result.append(label_list)
    print(label_result)
    # CSVファイルにラベリング結果を書き出し
    write_csv(label_result, result_noun_path)


def object_recognition(result_cut_img_path, result_noun_path):
    # --------------------------------------------------
    # 動画IDリストの作成
    # --------------------------------------------------
    video_id_list = os.listdir(result_cut_img_path)  # 動画ファイル名（動画ID）一覧を取得
    
    # --------------------------------------------------
    # 学習済みモジュールの読み込み
    # --------------------------------------------------
    print('モジュールの読み込み中...')
    # モジュールの読み込み
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures['default']
    print('モジュール読み込み完了')
    # TODO 読み取りエラーの例外処理追加

    # --------------------------------------------------
    # 物体認識によるラベル付け
    # --------------------------------------------------
    label_result = []   # 最終結果用のラベルリスト
    for video_id in video_id_list:
        print('==========')
        print(video_id)
        print('==========')
        cut_img_path = result_cut_img_path + '\\' + video_id # 保存先パス

        # ファイルの読み込み
        files = os.listdir(cut_img_path)
        label_list = []     # ラベルのリスト

        for i in range(len(files)):
            path = cut_img_path + '\\cut_img' + str(i+1) + '.jpg'  # カットのパス
            print(path)
            cut_no = 'cut_' + str(i+1)  # カット番号
            print('[' + cut_no + ']')

            # 物体検出・認識
            label = run_detector(detector, path)            # 検出結果のラベル
            label_list.append([video_id, cut_no, label])    # [動画ID, カット番号, ラベルリスト]

            print('-----------------')
        label_result.append(label_list)
    print(label_result)
    # CSVファイルにラベリング結果を書き出し
    write_csv(label_result, result_noun_path)

if __name__ == '__main__':
    start = time.time()
    #exe()
    object_recognition(r'C:\Users\fukuyori\Result\Cut_Img', r'C:\Users\fukuyori\OneDrive\デスクトップ\研究\CM_Analysis\noun.csv')
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
