import os
import ast
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from logging import getLogger, StreamHandler, DEBUG
import cv2

from utils.file_io import read_csv, write_csv, create_dest_folder
from cut_segmentation_mod import read_video_data

# ログ設定
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

#文書ベクトル化関数
def vecs_array(documents):
    from sklearn.feature_extraction.text import TfidfVectorizer
    np.set_printoptions(precision=2)
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(use_idf=False, token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(docs)

    return vecs.toarray()

def equal_list(lst1, lst2):
    lst = lst1.copy()
    for element in lst2:
        try:
            lst.remove(element)
        except ValueError:
            break
    else:
        if not lst:
            return True
    return False

def subtract_list(lst1, lst2):
    lst = lst1.copy()
    for element in lst2:
        try:
            lst.remove(element)
        except ValueError:
            pass
    return lst

def intersect_list(lst1, lst2):
    arr = []
    lst = lst1.copy()
    for element in lst2:
        try:
            lst.remove(element)
        except ValueError:
            pass
        else:
            arr.append(element)
    return arr

# ラベル同士の比較、類似度の算出
def label_comp(prev_id, next_id, prev_label, next_label):
    # 両方のラベルデータが存在するとき
    if prev_label and next_label:  
        # 両ラベルの類似度算出
        docs = [' '.join(prev_label), ' '.join(next_label)] # 2つの文書にする
        cs_array = np.round(cosine_similarity(vecs_array(docs), vecs_array(docs)),3)    # 2つの文書から類似行列を算出
        similarity = cs_array[0][1] # 類似行列から類似度を抽出

        # 類似度が閾値以上の時
        if similarity >= 0.93 and prev_id == next_id:
            return True
        else:
            return False
    else:
        return False

# シーン分割点を計算
def calc_scene_point(scene_data):
    current_id = scene_data[0][0]   # 現在の動画ID
    scene_point_dic = {}    # シーン分割点の辞書
    l = []
    for data in scene_data:
        video_id = data[0]  # 動画ID
        end = int(data[3])  # エンドフレーム

        if current_id != video_id:
            scene_point_dic[current_id] = l
            current_id = video_id
            l = []
        
        l.append(end)

        if scene_data.index(data) == len(scene_data)-1:
            scene_point_dic[current_id] = l
    
    #print(scene_point_dic)
    return scene_point_dic   

# シーンを保存する
def save_scene(scene_point_dic):
    base = os.path.dirname(os.path.abspath(__file__))   # スクリプト実行ディレクトリ
    #result_scene_path = os.path.normpath(os.path.join(base, r'result\Scene'))  # シーン保存ディレクトリ
    result_scene_path = os.path.normpath(os.path.join(base, r'result\Scene_1'))  # シーン保存ディレクトリ
    video_path = os.path.normpath(os.path.join(base, r'data\movie'))    # 動画パス
    
    # シーン保存先のフォルダを作成
    create_dest_folder(result_scene_path)

    # --------------------------------------------------
    # 動画IDリストの作成
    # --------------------------------------------------
    files = os.listdir(video_path)  # 動画ファイル名（動画ID）一覧を取得
    video_id_list = [f.replace('.mp4', '') for f in files]  # 一覧から拡張子を削除

    # --------------------------------------------------
    # シーン分割
    # --------------------------------------------------
    for video_id in video_id_list:
        input_video_path = video_path + '\\' + video_id + '.mp4' # 動画ファイルの入力パス 
        scene_point = scene_point_dic[video_id] # シーンの分割点

        # --------------------------------------------------
        # 保存先フォルダの作成
        # --------------------------------------------------
        dest_path = os.path.join(result_scene_path, video_id) # 各動画のカット分割結果の保存先
        create_dest_folder(dest_path)   # フォルダ作成 
        
        # --------------------------------------------------
        # 動画の読み込み、フレームデータと動画情報を抽出
        # --------------------------------------------------
        frames, video_info = read_video_data(input_video_path)
                
        # --------------------------------------------------
        # シーン分割点の情報を使用して、動画を分割して保存
        # --------------------------------------------------
        fps, width, height = video_info # 動画情報の展開

        #fourcc = cv2.VideoWriter_fourcc('m','p','4','v')    # 動画の保存形式
        fourcc = 0x00000021    # 動画の保存形式(H264形式でエンコード)　2021/2/12 変更
        writer = [] # 書き込み用のリスト
        begin = 0   # シーン最初のフレーム

        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]   # RGBからBGRに戻す（戻さないと色が反転したまま保存される）

        scene_count = len(scene_point) # シーン数
        for i in range(scene_count):
            writer.append(cv2.VideoWriter(dest_path + '\\scene' + str(i+1) + '.mp4', fourcc, fps, (int(width), int(height))))
            for j in range(begin, scene_point[i]+1):
                writer[i].write(frames[j])
            begin = scene_point[i]+1

        logger.debug(video_id + '_scene1 ～ ' + str(scene_count) + 'を保存しました')
        logger.debug('保存先 : ' + dest_path)
        logger.debug('-' * 90)

# シーンの統合
def scene_integration():
    base = os.path.dirname(os.path.abspath(__file__))   # スクリプト実行ディレクトリ
    cut_data_path = os.path.normpath(os.path.join(base, r'result\Label\merged_label.csv'))  # カットデータのパス
    scene_data_path = os.path.normpath(os.path.join(base, r'result\Label\result_label.csv'))  # カットデータのパス
    
    # カットデータの読み込み　
    # ['動画ID', 'カット番号', 'スタートフレーム', 'エンドフレーム', '[ラベルのリスト]']
    cut_data = read_csv(cut_data_path, True)    # カットデータ 

    # シーンデータ作成 
    # ['動画ID', 'シーン番号', 'スタートフレーム', 'エンドフレーム', '[ラベルのリスト]']
    scene_data = [] 
    for i in range(len(cut_data)):
        # インデックスが最後の場合、類似度は算出せずにシーンデータに追加
        if i == len(cut_data)-1:
            label = [l for l in ast.literal_eval(cut_data[i][4])]  # ラベルデータ

            scene_data.append([cut_data[i][0], cut_data[i][2], cut_data[i][3], label])
            break

        prev_id = cut_data[i][0]      # 動画ID
        next_id = cut_data[i+1][0]    # 次データの動画ID

        prev_label_data = ast.literal_eval(cut_data[i][4])   # ラベル
        next_label_data = ast.literal_eval(cut_data[i+1][4]) # 次データのラベル
    
        prev_label = [labels[0] if isinstance(labels, list) else labels for labels in prev_label_data] # ラベル
        next_label = [labels[0] if isinstance(labels, list) else labels for labels in next_label_data] # 次データのラベル

        cut_range = (cut_data[i][2], cut_data[i][3])    # カット範囲

        # 統合するかを判定（True : 統合する，False : 統合しない）
        isintegrate = label_comp(prev_id, next_id, prev_label, next_label)
        if isintegrate:
            label = prev_label_data + next_label_data

            print(prev_id, cut_data[i][1], label)  # シーン統合箇所（TODO: カット画像が複数になるため、アノテーションが困難か？）
            cut_range = (cut_data[i][2], cut_data[i+1][3])  # カット範囲統合
        else:
            label = prev_label_data
                
        # ラベルデータに追加
        scene_data.append([prev_id, cut_range[0], cut_range[1], label])

    # カット範囲が重なっているデータを削除する
    remove_target = [n for p, n in zip(scene_data, scene_data[1:]) if p[0] == n[0] and int(p[2]) >= int(n[1])]
    for rt in remove_target:
        scene_data.remove(rt)

    scene_no_list = []
    for i in range(len(scene_data)):
        # 最初のインデックスの時
        if i == 0:
            scene_no = 1
        else:
            # 次データのidと同じとき
            if scene_data[i][0] == scene_data[i-1][0]:
                scene_no += 1
            else:
                scene_no = 1

        scene_no_list.append(scene_no)

    for data, scene_no in zip(scene_data, scene_no_list):
        data.insert(1, 'scene_' + str(scene_no)) 

    # シーン分割点の辞書を作成
    scene_point_dic = calc_scene_point(scene_data)

    # シーンを動画として新たに保存
    save_scene(scene_point_dic)

    # 見出しを追加して、シーンデータをCSVファイルに保存
    scene_data.insert(0, ['動画ID', 'シーン番号', 'スタートフレーム', 'エンドフレーム', '[ラベルのリスト]'])
    
    write_csv(scene_data, scene_data_path)

scene_integration()