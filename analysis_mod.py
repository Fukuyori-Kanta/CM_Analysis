import csv
import random
import cv2
import sys
import os
import shutil
import ast
from collections import Counter
import heapq
import statistics
import numpy as np

def getEncode(filepath):
    encs = "iso-2022-jp euc-jp sjis utf-8-sig".split()
    for enc in encs:
        with open(filepath, encoding=enc) as fr:
            try:
                fr = fr.read()
            except UnicodeDecodeError:
                continue
        return enc

def read_favo(file_path, needs_skip_header=False):
    """
    CSVファイルを読み込んで、その結果を返す関数

    Parameters
    ----------
    file_path : str
        読み込むCSVファイルのパス

    needs_skip_header : bool, default False
        ヘッダーを読み飛ばすかどうか
        
    Returns
    -------
    l : list
        読み込んだ結果を返すリスト
    """
    # 文字コードを取得
    enc = getEncode(file_path)

    csvfile = open(file_path, 'r', encoding=enc)
    reader = csv.reader(csvfile, delimiter=',')

    # ヘッダーを読み飛ばしたい時
    if needs_skip_header:
        header = next(reader)  

    l = []
    for row in reader:
        l.append(row)
    
    return l

def read_csv(file_path, needs_skip_header=False):
    """
    CSVファイルを読み込んで、その結果を返す関数

    Parameters
    ----------
    file_path : str
        読み込むCSVファイルのパス

    needs_skip_header : bool, default False
        ヘッダーを読み飛ばすかどうか
        
    Returns
    -------
    l : list
        読み込んだ結果を返すリスト
    """
    # 文字コードを取得
    enc = getEncode(file_path)

    csvfile = open(file_path, 'r', encoding=enc)
    reader = csv.reader(csvfile)

    # ヘッダーを読み飛ばしたい時
    if needs_skip_header:
        header = next(reader)  

    l = []
    for row in reader:
        l.append(row)
    
    return l
    
def write_csv(data, dest_path):
    """
    データを受け取り、CSVに書き出す関数

    Parameters
    ----------
    data : list
        出力するデータ
        
    dest_path : str
        保存先フォルダのパス
    """
    with open(dest_path, 'w', newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        if len(data) >= 2: # 2次元配列以上の場合
            writer.writerows(data)

        else:   # 1次元配列の場合
            writer.writerow(data)

def result_write_csv(data, dest_path, i):
    """
    データを受け取り、CSVに書き出す関数

    Parameters
    ----------
    data : list
        出力するデータ
        
    dest_path : str
        保存先フォルダのパス
    """
    if i == 0:
        mode = 'w'
    else:
        mode = 'a'

    with open(dest_path, mode, newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        writer.writerows(data)
        """
        if len(data) >= 2: # 2次元配列以上の場合
            writer.writerows(data)

        else:   # 1次元配列の場合
            writer.writerow(data)
        """
def topN(dic, N):
    # valueでソートしたときの上位N個
    lists = heapq.nlargest(N, dic.items(), key=lambda x: x[1])

    new_dict = {}
    for l in lists:
        new_dict[l[0]] = l[1]
    
    return new_dict

def bottomN(dic, N):
    # valueでソートしたときの上位N個
    lists = heapq.nsmallest(N, dic.items(), key=lambda x: x[1])

    new_dict = {}
    for l in lists:
        new_dict[l[0]] = l[1]
    
    return new_dict

def data_shaping(scene_data, favo_data, video_id_list):
    """
    docstring
    """
    # データの整形（動画IDをキーとした辞書の作成）
    # シーンデータの整形
    # {video_id : [scene_no, start, end, label], [scene_no, start, end, label], ...}
    prev_id = scene_data[0][0]   # 前データのID
    scene_dic = {}  # 動画IDごとにシーンデータを辞書化
    l = []
    for data in scene_data:
        video_id = data[0]      # 動画ID
        scene_no = data[1]      # シーン番号
        start = int(data[2])    # スタートフレーム
        end = int(data[3])      # エンドフレーム
        label = ast.literal_eval(data[4])   # ラベルのリスト

        # 前デーのIDと動画IDが違うとき
        if prev_id != video_id:
            scene_dic[prev_id] = l  # 
            prev_id = video_id
            l = []
        
        l.append([scene_no, start, end, label])

        # 最後のデータ
        if scene_data.index(data) == len(scene_data)-1:
            scene_dic[prev_id] = l
    
    # 好感度データの整形
    # {video_id : [sec, favo], [sec, favo], ...}
    favo_dic = {}   # 動画IDごとに好感度データを辞書化
    l = []
    for data in favo_data:
        #data = ast.literal_eval(data)
        video_id = data[3]      # 動画ID
        sec = int(data[4])      # 経過秒数
        favo = float(data[6])   # 好感度

        if video_id in video_id_list:
            l.append([sec, favo])

            # 経過秒数が15の時
            if sec == 15:
                favo_dic[video_id] = l
                l = []
    
    return scene_dic, favo_dic

def favo_facter_data_shaping(scene_data, favo_data, video_id_list, i):
    """
    docstring
    """
    # データの整形（動画IDをキーとした辞書の作成）
    # シーンデータの整形
    # {video_id : [scene_no, start, end, label], [scene_no, start, end, label], ...}
    prev_id = scene_data[0][0]   # 前データのID
    scene_dic = {}  # 動画IDごとにシーンデータを辞書化
    l = []
    for data in scene_data:
        video_id = data[0]      # 動画ID
        scene_no = data[1]      # シーン番号
        start = int(data[2])    # スタートフレーム
        end = int(data[3])      # エンドフレーム
        label = ast.literal_eval(data[4])   # ラベルのリスト

        # 前デーのIDと動画IDが違うとき
        if prev_id != video_id:
            scene_dic[prev_id] = l  # 
            prev_id = video_id
            l = []
        
        l.append([scene_no, start, end, label])

        # 最後のデータ
        if scene_data.index(data) == len(scene_data)-1:
            scene_dic[prev_id] = l
    
    # 好感度データの整形
    # {video_id : [sec, favo], [sec, favo], ...}
    favo_dic = {}   # 動画IDごとに好感度データを辞書化
    l = []
    for data in favo_data:
        video_id = data[3]      # 動画ID
        sec = int(data[4])      # 経過秒数
        favo = float(data[5+i])   # 好感要因

        if video_id in video_id_list:
            l.append([sec, favo])

            # 経過秒数が15の時
            if sec == 15:
                favo_dic[video_id] = l
                l = []
    
    return scene_dic, favo_dic

def get_best_favo(start, end, favo_list):
    """
    シーン範囲に近い秒数の好感度を返す関数
    """
    sec_list = [i*30 for i in range(1, 16)]
    start_near_sec = sec_list[np.abs(np.asarray(sec_list) - start).argmin()]
    end_near_sec = sec_list[np.abs(np.asarray(sec_list) - end).argmin()]
    near_sec_list =  [start_near_sec+i*30 for i in range((end_near_sec-start_near_sec)//30+1)]
    candidate_favo = [favo_list[ns//30-1][1] for ns in near_sec_list]
    
    # 好感度の平均値を返す
    return sum(candidate_favo) / len(candidate_favo)

def ranking(scene_favo_dic, scene_dic, i, result_path):

    # 上位、中位、下位でシーンに付与されたラベル上位10件を出力
    dic_len = len(scene_favo_dic)   # 総シーン数
    all_favo = topN(scene_favo_dic, dic_len)   # 好感度の降順

    #all_favo_list = []
    l = []
    for scene in all_favo.keys():
        video_id = scene[0]
        scene_no = scene[1]
        favo_val = all_favo[scene]
        
        # 該当のラベルデータ読み出し TODO (id, no)をキーとしたラベルデータの作成
        label = [sd[3] for sd in scene_dic[video_id] if sd[0] == scene_no][0]
        
        l.append([video_id, scene_no, favo_val, label])
    
    # 上位50シーン
    # とりあえず、60シーンで行い、ノイズになっている部分を目視確認し、減算
    top_scene = l[:1000]
    top_label = []
    for scene in top_scene:
        label = [t[:-3] for t in scene[3]] # 個数削除
        #label = [t for t in scene[3]] # 個数削除
        top_label += label
        
    #top_cnt = [c[0] + '(' + str(c[1]) + ')' for c in Counter(top_label).most_common()]
    top_cnt = [[c[0], str(c[1])] for c in Counter(top_label).most_common()]
    
    for t in top_cnt[:10]:
        print(t)
    print('-----------')

    # 中位
    mid_scene = l[dic_len//2-500:dic_len//2+500]
    mid_label = []
    for scene in mid_scene:
        label = [t[:-3] for t in scene[3]] # 個数削除
        #label = [t for t in scene[3]] # 個数削除
        mid_label += label

    #mid_cnt = [c[0] + '(' + str(c[1]) + ')' for c in Counter(mid_label).most_common()]
    mid_cnt = [[c[0], str(c[1])]   for c in Counter(mid_label).most_common()]
    for m in mid_cnt[:10]:
        print(m)
    print('-----------')

    # 下位
    btm_scene = l[-1000:]
    btm_label = []
    for scene in btm_scene:
        label = [t[:-3] for t in scene[3]] # 個数削除
        #label = [t for t in scene[3]] # 個数削除
        btm_label += label

    #btm_cnt = [c[0] + '(' + str(c[1]) + ')' for c in Counter(btm_label).most_common()]
    btm_cnt = [[c[0], str(c[1])]   for c in Counter(btm_label).most_common()]

    for b in btm_cnt[:10]:
        print(b)

    # CSVファイルに出力
    data = [[['上位ラベル名', '個数']], top_cnt[:10], [['---------', '---------']], 
            [['中位ラベル名', '個数']], mid_cnt[:10], [['---------', '---------']], 
            [['下位ラベル名', '個数']], btm_cnt[:10]]
    for num in range(len(data)):
        if i == 0:
            file_name = 'facter_' + str(i) + '.csv'
        elif i == 1:
            file_name = 'facter_A.csv'
        else:
            file_name = 'facter_' + str(i-1) + '.csv'
        
        output_path = os.path.normpath(os.path.join(result_path, file_name))
        result_write_csv(data[num], output_path, num)
        
def main():
    base = os.path.dirname(os.path.abspath(__file__))   # スクリプト実行ディレクトリ    
    scene_path = os.path.normpath(os.path.join(base, r'Result\Label\result_label.csv'))  # シーンデータのパス
    favo_path = os.path.normpath(os.path.join(base, r'Data\好感度データ\favorability_data.csv'))  # 好感度データのパス
    video_path = os.path.normpath(os.path.join(base, r'Data\Movie'))  # 動画データのパス
    result_path = os.path.normpath(os.path.join(base, r'Result\Favo'))  # 動画データのパス

    # 場面データの読み込み
    scene_data = read_csv(scene_path, True)

    # 好感度データの読み込み
    favo_data = read_favo(favo_path, True)

    # 動画IDリストの作成
    files = os.listdir(video_path)  # 動画ファイル名（動画ID）一覧を取得
    video_id_list = [f.replace('.mp4', '') for f in files]  # 一覧から拡張子を削除
    
    """
    # データの整形
    scene_dic, favo_dic = data_shaping(scene_data, favo_data, video_id_list)

    # 各シーンに好感度を付与
    # {(video_id, scene_no) : favo, ((video_id, scene_no) : favo, ...}
    scene_favo_dic = {} # 各シーンの好感度 
    for video_id in video_id_list:
        favo_list = favo_dic[video_id]
        scene_list = scene_dic[video_id]

        for sl in scene_list:
            scene_no = sl[0]
            start = sl[1]
            end = sl[2]
            label = sl[3]
            
            # シーン範囲から最適な好感度を取得
            scene_favo_dic[(video_id, scene_no)] = get_best_favo(start, end, favo_list)
    
    # 上位、中位、下位でシーンに付与されたラベル上位10件を出力
    ranking(scene_favo_dic, scene_dic, 0, result_path)

    # 出力
    scene_favo_list = [[sfd[0], sfd[1], scene_favo_dic[sfd]] for sfd in scene_favo_dic.keys()]

    write_csv(scene_favo_list, r'C:\Users\fukuyori\CM_Analysis\Result\Favo\favo.csv')
    
    """
    #################################
    # 好感度要因15項目
    #################################
    for i in range(17):
        if i == 0:
            print('------- 好感度 -------')
        elif i == 1:
            print('------- A -------')
        else:
            print('------- R' + str(i-1) + '-------')

        # データの整形
        scene_dic, favo_dic = favo_facter_data_shaping(scene_data, favo_data, video_id_list, i)

        # 各シーンに好感度を付与
        # {(video_id, scene_no) : favo, ((video_id, scene_no) : favo, ...}
        scene_favo_dic = {} # 各シーンの好感度 
        for video_id in video_id_list:
            favo_list = favo_dic[video_id]
            scene_list = scene_dic[video_id]

            for sl in scene_list:
                scene_no = sl[0]
                start = sl[1]
                end = sl[2]
                label = sl[3]
                
                # シーン範囲から最適な好感度を取得
                scene_favo_dic[(video_id, scene_no)] = get_best_favo(start, end, favo_list)
        # データの整形下位でシーンに付与されたラベル上位10件を出力
        ranking(scene_favo_dic, scene_dic, i, result_path)
    
if __name__=="__main__":
    main()
