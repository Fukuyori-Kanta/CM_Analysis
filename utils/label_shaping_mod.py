import json
import os 
import ast
import csv
import operator
from utils.file_io import read_txt, read_csv, write_csv

# ラベルの翻訳辞書作成
def trans_dic_creation(label_list_en, label_list_ja):
    label_trans_dic = {}    # ラベルの翻訳辞書 {english : japanese}
    for en, ja in zip(label_list_en, label_list_ja):
        label_trans_dic[en] = ja

    return label_trans_dic

# 名詞ラベルの整形
def noun_label_shaping():
    base_path = os.path.dirname(os.path.abspath(__file__))  # スクリプト実行ディレクトリ
    noun_result_path = os.path.normpath(os.path.join(base_path, r'result\Label\noun_label.csv'))   # 物体認識のラベル結果（名詞ラベル）
    video_path = os.path.normpath(os.path.join(base_path, r'data\movie'))                          # 動画データのパス
    noun_en_path = os.path.normpath(os.path.join(base_path, r'data\Label\noun_label(en).txt'))     # データセットのラベル群（英語）
    noun_ja_path = os.path.normpath(os.path.join(base_path, r'data\Label\noun_label(ja).txt'))     # データセットのラベル群（日本語）
    trans_table_path = os.path.normpath(os.path.join(base_path, r'data\Label\noun_screening.csv')) # スクリーニング用の対応表
        
    # ラベル一覧を読み込み（英語、日本語）
    noun_label_en = read_txt(noun_en_path)
    noun_label_ja = read_txt(noun_ja_path)
    
    # ラベルの翻訳辞書作成
    label_trans_dic = trans_dic_creation(noun_label_en, noun_label_ja)
    
    # スクリーニング用データ
    trans_table = {data[0]:data[1] for data in read_csv(trans_table_path)}

    # 名詞ラベル結果を取得
    noun_result = [ast.literal_eval(l) for result in read_csv(noun_result_path) for l in result]
    
    noun_label = []  # 整形後の名詞ラベル結果
    for result in noun_result:
        labels = result[2]   # ラベルデータ
        
        # 動画リストとラベル結果の動画IDが同じの時、
        # ラベル名を翻訳・スクリーニングして結果に格納する
        for label in labels:
            # 英語から日本語へ翻訳
            label_name_en = label[0]
            label_name_ja = label_trans_dic[label_name_en]

            # スクリーニング
            if label_name_ja in trans_table: # 変換辞書に存在すれば
                label_name_ja = trans_table[label_name_ja]
            
            # 結果を書き換える
            label[0] = label_name_ja
            
        noun_label.append(result)

    return noun_label

# 動詞ラベルの整形
def verb_label_shaping():
    base_path = os.path.dirname(os.path.abspath(__file__))  # スクリプト実行ディレクトリ
    verb_result_path = os.path.normpath(os.path.join(base_path, r'result\Label\verb_label.json'))  # 動作認識のラベル結果（名詞ラベル）
    verb_en_path = os.path.normpath(os.path.join(base_path, r'data\Label\verb_label(en).txt'))     # データセットのラベル群（英語）
    verb_ja_path = os.path.normpath(os.path.join(base_path, r'data\Label\verb_label(ja).txt'))     # データセットのラベル群（日本語）
    trans_table_path = os.path.normpath(os.path.join(base_path, r'data\Label\verb_screening.csv')) # スクリーニング用の対応表

    # ラベル一覧を読み込み（英語、日本語）
    verb_label_en = read_txt(verb_en_path)
    verb_label_ja = read_txt(verb_ja_path)
    
    # ラベルの翻訳辞書作成
    label_trans_dic = trans_dic_creation(verb_label_en, verb_label_ja)

    # スクリーニング用データ
    trans_table = {data[0]:data[1] for data in read_csv(trans_table_path)}

    # 動詞ラベル結果を取得
    json_open = open(verb_result_path, 'r')
    verb_result = json.load(json_open)

    # マッチングしたら、翻訳
    verb_label = []
    for result in verb_result:
        video_id = result['video'].replace('.mp4', '').split('_')[0]    # 動画ID
        cut_no = result['video'].replace('.mp4', '').split('_')[1]      # カット番号
        cut_no = cut_no[:3] + '_' + cut_no[3:] # cut〇 → cut_〇 に変更

        l = []  # 仮の格納場所
        l.append(video_id)
        l.append(cut_no)

        cut_label = []
        for c in result['clips']:
            if max(c['scores']) >= 7.0:
                # スクリーニング
                if label_trans_dic[c['label']] in trans_table: # 変換辞書に存在すれば
                    cut_label.append(trans_table[label_trans_dic[c['label']]])
                    
        l.append(list(set(cut_label)))
        verb_label.append(l)

    return verb_label
    
def label_shaping(noun_label_path, verb_label_path, label_path):
    noun_label = [[data[0], data[1], ast.literal_eval(data[2])] for data in read_csv(noun_label_path, needs_skip_header=True)]
    verb_label = [[data[0], data[1], ast.literal_eval(data[2])] for data in read_csv(verb_label_path, needs_skip_header=True)]

    if len(noun_label) != len(verb_label):
        raise ValueError('物体検出と動作認識のレコード数が異なっています。')

    result_label = [{'video_id': noun[0], 'cut_no': int(noun[1]), 'labels': noun[2] + verb[2]} for noun, verb in zip(noun_label, verb_label)]

    # 結果をソート
    label_results = sorted(label_results, key=operator.itemgetter('video_id', 'cut_no'))
    
    # CSVファイルに保存
    field_name = ['video_id', 'cut_no', 'labels']
    with open(label_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_name)
        writer.writeheader()
        writer.writerows(result_label)