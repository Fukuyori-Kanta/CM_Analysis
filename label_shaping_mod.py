import json
import os 
import ast
import re

from file_io import read_csv, write_csv

# テキストファイルの読み込み
def read_txt(path):
    with open(path, encoding='utf-8') as f:
        l = [s.strip() for s in f.readlines()]

    return l

# ラベルの翻訳辞書作成
def trans_dic_creation(label_list_en, label_list_ja):
    label_trans_dic = {}    # ラベルの翻訳辞書 {english : japanese}
    for en, ja in zip(label_list_en, label_list_ja):
        label_trans_dic[en] = ja

    return label_trans_dic

# 名詞ラベルの整形
def noun_label_shaping():
    base_path = os.path.dirname(os.path.abspath(__file__))  # スクリプト実行ディレクトリ
    noun_result_path = os.path.normpath(os.path.join(base_path, r'Result\Label\noun_label.csv'))   # 物体認識のラベル結果（名詞ラベル）
    video_path = os.path.normpath(os.path.join(base_path, r'Data\Movie'))                          # 動画データのパス
    noun_en_path = os.path.normpath(os.path.join(base_path, r'Data\Label\noun_label(en).txt'))     # データセットのラベル群（英語）
    noun_ja_path = os.path.normpath(os.path.join(base_path, r'Data\Label\noun_label(ja).txt'))     # データセットのラベル群（日本語）
    trans_table_path = os.path.normpath(os.path.join(base_path, r'Data\Label\noun_screening.csv')) # スクリーニング用の対応表
    
    # 動画リストの作成
    files = os.listdir(video_path)  # 動画ファイル名（動画ID）一覧を取得
    video_id_list = [f.replace('.mp4', '') for f in files]  # 一覧から拡張子を削除
    
    # ラベル一覧を読み込み（英語、日本語）
    noun_label_en = read_txt(noun_en_path)
    noun_label_ja = read_txt(noun_ja_path)
    
    # ラベルの翻訳辞書作成
    label_trans_dic = trans_dic_creation(noun_label_en, noun_label_ja)
    
    # スクリーニング用データ
    trans_table = {data[0]:data[1] for data in read_csv(trans_table_path)}

    # 名詞ラベル結果を取得
    noun_result = [ast.literal_eval(l) for l in read_csv(noun_result_path)[0]]

    noun_label = []
    for video_id in video_id_list:
        for result in noun_result:
            # 動画リストとラベル結果の動画IDが同じなら
            if video_id == result[0][0]:
                for i in range(len(result)):
                    for j in range(len(result[i][2])):
                        
                        # スコア部分を削除
                        l_en = re.sub('\([0-9]*\.[0-9]*%\)', '', result[i][2][j]) 

                        # 英語から日本語へ翻訳
                        result[i][2][j] = label_trans_dic[l_en]  

                        # スクリーニング
                        if result[i][2][j] in trans_table: # 変換辞書に存在すれば
                            result[i][2][j] = trans_table[result[i][2][j]]
                        
                    noun_label.append(result[i])

    return noun_label

# 動詞ラベルの整形
def verb_label_shaping():
    base_path = os.path.dirname(os.path.abspath(__file__))  # スクリプト実行ディレクトリ
    verb_result_path = os.path.normpath(os.path.join(base_path, r'Result\Label\verb_label.json'))  # 動作認識のラベル結果（名詞ラベル）
    verb_en_path = os.path.normpath(os.path.join(base_path, r'Data\Label\verb_label(en).txt'))     # データセットのラベル群（英語）
    verb_ja_path = os.path.normpath(os.path.join(base_path, r'Data\Label\verb_label(ja).txt'))     # データセットのラベル群（日本語）
    trans_table_path = os.path.normpath(os.path.join(base_path, r'Data\Label\verb_screening.csv')) # スクリーニング用の対応表

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

# 動詞ラベルと名詞ラベルの結合
def noun_verb_merge(noun_label, verb_label):
    
    # 名詞ラベルをベースに、動作ラベルが付与されているならば、両ラベルを結合
    for noun in noun_label:
        for verb in verb_label:
            # 動画IDとカット番号が一致するときに結合
            if noun[0] == verb[0] and noun[1] == verb[1]:
                noun[2] += verb[2]  # 動詞ラベルと名詞ラベルを結合して格納
            
    return noun_label

#カット範囲の追加
def cut_range_add(label, cut_point_list):
    
    cut_point_list = [ast.literal_eval(cut_point) for cut_point in cut_point_list[1]]
    cut_range_list = [] # カット範囲のリスト
    
    for cut_point in cut_point_list:
        begin = 0   # カット最初のフレーム
        cut_count = len(cut_point) # カット数

        for i in range(cut_count):
            cut_range_list.append(tuple([begin, cut_point[i]]))
            begin = int(cut_point[i])+1

    label_data = [] # ラベルデータ

    # 見出しの追加
    label_data.append(['動画ID', 'カット番号', 'スタートフレーム', 'エンドフレーム', '[ラベルのリスト]'])   

    # データの追加
    for l, cut_range in zip(label, cut_range_list):
        l.insert(2, cut_range[0])
        l.insert(3, cut_range[1])
        label_data.append(l)

    return label_data

def label_shaping():
    """
    ['動画ID', 'カット番号', 'スタートフレーム', 'エンドフレーム', '[ラベルのリスト]']

    """
    base_path = os.path.dirname(os.path.abspath(__file__))  # スクリプト実行ディレクトリ
    cut_point_path = os.path.normpath(os.path.join(base_path, r'Result\cut_point.csv'))   # 動作認識のラベル結果（名詞ラベル）
    merged_label_path = os.path.normpath(os.path.join(base_path, r'Result\Label\merged_label.csv'))   # 動作認識のラベル結果（名詞ラベル）
    
    # 名詞ラベルの整理
    noun_label = noun_label_shaping()

    # 動詞ラベルの整理
    verb_label = verb_label_shaping()

    # 名詞ラベルと動詞ラベルの結合
    label = noun_verb_merge(noun_label, verb_label)
    
    # カット点のリストを読み込み
    cut_point_list = read_csv(cut_point_path)
    label = cut_range_add(label, cut_point_list)   # カット範囲情報を追加

    # CSVファイルに書き出し
    write_csv(label, merged_label_path)