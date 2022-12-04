import os
import ast
from collections import Counter
import heapq
import numpy as np
import re

from utils.file_io import read_csv, read_favo, write_ranking_data, write_csv

def order_desc(dic, N):
    # valueでソートしたときの上位N個
    lists = heapq.nlargest(N, dic.items(), key=lambda x: x[1])

    new_dict = {}
    for l in lists:
        new_dict[l[0]] = l[1]
    
    return new_dict

def shaping_favo_data(scene_data, favo_data, video_id_list, i):
    """
    好感度データを整形して返す関数
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
        label = data[4]   # ラベルのリスト

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
    return round(sum(candidate_favo) / len(candidate_favo), 5)

def ranking(scene_favo_dic, scene_dic, i, result_path):
    # 上位、中位、下位で〇〇シーンに付与されたラベル上位10件を出力
    dic_len = len(scene_favo_dic)   # 総シーン数
    all_favo = order_desc(scene_favo_dic, dic_len)   # 好感度の降順
    scene_range = 1000  # 1000件のシーン範囲

    all_scene = []  # 全シーンのデータ [動画名, シーン番号, 好感度, ラベルのリスト]
    for scene in all_favo.keys():
        video_id = scene[0]
        scene_no = scene[1]
        favo_val = all_favo[scene]
        
        # 該当のラベルデータ読み出し
        label = [sd[3] for sd in scene_dic[video_id] if sd[0] == scene_no][0]
        
        all_scene.append([video_id, scene_no, favo_val, label])
    
    all_label = []  # 全ラベルのデータ
    for scene in all_scene:
        label = [re.sub("\(.*\)", "", s) for s in scene[3]] # 個数削除
        all_label += label
        
    # 総ラベル件数辞書 {'女性' : 3112, '男性' : 2362, ...}
    all_cnt = [[c[0], str(c[1])] for c in Counter(all_label).most_common()]
    label_cnt_dic = {a[0]:int(a[1]) for a in all_cnt}
    
    # 上位
    top_scene = all_scene[:scene_range]
    top_label = []
    for scene in top_scene:
        label = [re.sub("\(.*\)", "", s) for s in scene[3]] # 個数削除
        top_label += label

    # 件数順にソート 
    top_cnt = [[c[0], c[1], str(round(c[1] / label_cnt_dic[c[0]], 2))] for c in Counter(top_label).most_common()]
    
    for t in top_cnt[:10]:
        print(t)
    print('-----------')
    
    # 中位
    mid_scene = all_scene[(dic_len//2)-(scene_range//2):(dic_len//2)+(scene_range//2)]
    mid_label = []
    for scene in mid_scene:
        label = [re.sub("\(.*\)", "", s) for s in scene[3]] # 個数削除
        mid_label += label
    
    # 件数順にソート
    mid_cnt = [[c[0], c[1], str(round(c[1] / label_cnt_dic[c[0]], 2))]   for c in Counter(mid_label).most_common()]

    for m in mid_cnt[:10]:
        print(m)
    print('-----------')

    # 下位
    btm_scene = all_scene[-scene_range:]
    btm_label = []
    for scene in btm_scene:
        label = [re.sub("\(.*\)", "", s) for s in scene[3]] # 個数削除
        btm_label += label

    # 件数順にソート
    btm_cnt = [[c[0], c[1], str(round(c[1] / label_cnt_dic[c[0]], 2))]   for c in Counter(btm_label).most_common()]

    for b in btm_cnt[:10]:
        print(b)
    
    # CSVファイルに出力
    data = [top_cnt[:10], mid_cnt[:10], btm_cnt[:10]]
    for num in range(len(data)):
        if i == 0:
            file_name = 'facter_' + str(i) + '.csv'
        elif i == 1:
            file_name = 'facter_A.csv'
        else:
            file_name = 'facter_' + str(i-1) + '.csv'
        
        output_path = os.path.normpath(os.path.join(result_path, file_name))
        write_ranking_data(data[num], output_path, num)

def favo_analysis():
    # パス設定
    base = os.path.dirname(os.path.abspath(__file__))   # スクリプト実行ディレクトリ    
    scene_path = os.path.normpath(os.path.join(base, r'result\Label\result_label.csv'))  # シーンデータのパス
    favo_path = os.path.normpath(os.path.join(base, r'data\好感度データ\favorability_data.csv'))  # 好感度データのパス
    label_path = os.path.normpath(os.path.join(base, r'result\Label\result_label.csv'))  # ラベルデータのパス
    video_path = os.path.normpath(os.path.join(base, r'data\movie'))    # 動画データのパス
    result_path = os.path.normpath(os.path.join(base, r'result\Favo'))  # 各好感要因の結果格納パス
    result_ALL_path = os.path.normpath(os.path.join(base, r'result\Favo\result_data_ALL.csv'))  # 全シーン，全好感要因の結果格納パス
    rank_ALL_path = os.path.normpath(os.path.join(base, r'result\Favo\rank_data_ALL.csv'))     # 全好感要因のランキング結果格納パス

    # 場面データの読み込み
    scene_data = read_csv(scene_path, True)

    # 好感度データの読み込み
    favo_data = read_favo(favo_path, True)

    # ラベルデータの読み込み
    label_data = read_csv(label_path, True)

    # 動画IDリストの作成
    files = os.listdir(video_path)  # 動画ファイル名（動画ID）一覧を取得
    video_id_list = [f.replace('.mp4', '') for f in files]  # 一覧から拡張子を削除
    
    result_ALL = [] # 全データ格納リスト

    #################################
    # 好感度, 試用意向, 15項目の好感要因でランキング作成
    #################################
    for i in range(17):
        if i == 0:
            print('------- 好感度 -------')
        elif i == 1:
            print('------- A -------')
        else:
            print('------- R' + str(i-1) + '-------')

        # データの整形
        scene_dic, favo_dic = shaping_favo_data(scene_data, favo_data, video_id_list, i)

        # 各シーンに好感度を付与
        # {(video_id, scene_no) : favo, ((video_id, scene_no) : favo, ...}
        scene_favo_dic = {}  # 各シーンの好感度 （辞書）
        scene_favo_list = [] # 各シーンの好感度（リスト）
        for video_id in video_id_list:
            favo_list = favo_dic[video_id]
            scene_list = scene_dic[video_id]

            for sl in scene_list:
                scene_no = sl[0]    # シーン番号
                start = sl[1]       # スタートフレーム
                end = sl[2]         # エンドフレーム
                
                # シーン範囲から最適な好感度を取得
                best_favo = get_best_favo(start, end, favo_list)

                scene_favo_dic[(video_id, scene_no)] = best_favo
                scene_favo_list.append([video_id, scene_no, best_favo])
                
        # データの整形下位でシーンに付与されたラベル上位10件を出力
        #ranking(scene_favo_dic, scene_dic, i, result_path)
        
        # 全データの格納、出力
        # 最初の処理で、result_ALLにラベルデータをコピーする
        # それ以降は各好感要因の数値を追加していく
        if i == 0:
            result_ALL = label_data

        for result_row, scene_favo in zip(result_ALL, scene_favo_list):
            result_row.append(scene_favo[2])
        
    # カラム名を追加
    result_ALL.insert(0, ['動画名', 'シーン番号', 'スタートフレーム', 'エンドフレーム', 'ラベルのリスト', 
                          '好感度', '試用意向', '出演者', 'ユーモラス', 'セクシー ', '宣伝文句', 
                          '音楽・サウンド', ' 商品にひかれた', '説得力に共感', 'ダサいけど憎めない', '時代の先端 ', 
                          '心がなごむ', 'ストーリー展開', '企業姿勢', '映像・画像', '周囲の評判 ', 'かわいらしい'])
    
    # CSVファイルに保存
    write_csv(result_ALL, result_ALL_path)

    # 全好感要因のランキング結果を1つの表にまとめる
    rank_data_ALL = []
    for i in range(17):
        if i == 0:
            file_name = 'facter_' + str(i) + '.csv'
            
        elif i == 1:
            file_name = 'facter_A.csv'
        else:
            file_name = 'facter_' + str(i-1) + '.csv'

        input_path = os.path.normpath(os.path.join(result_path, file_name))
        rank_data = read_csv(input_path)
        if i == 0:
            rank_data_ALL = rank_data

        else:
            for result_row, data in zip(rank_data_ALL, rank_data):
                result_row.append(data[0])
                result_row.append(data[1])
                result_row.append(data[2])

    write_csv(rank_data_ALL, rank_ALL_path)

favo_analysis()