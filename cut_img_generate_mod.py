import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import csv
from logging import getLogger, StreamHandler, DEBUG

from cut_segmentation_mod import dest_folder_create, read_video_data, cut_point_detect

# ログ設定
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

def cut_img_save(video_id, cut_point, frames, dest_path):
    """
    カット範囲から一番最後のフレームをカット画像として保存する
    
    一番最後のフレームの理由は、カット画像の調査（※）により、
    画像のブレが比較的少ないという結果を得たからである。

    ※カット画像の調査
        最適なカット画像を統計的に調査
        1. ランダムな15CMを決める
        2. 以下のカット画像を抽出（安定のフレームは既にやっているため、実質5種類）
            ・最初から0% のフレーム  (first)
            ・最初から25% のフレーム (quarter)
            ・最初から50% のフレーム (center)
            ・最初から75% のフレーム (hree_quarters)
            ・最初から100% のフレーム(last)
            ・安定のフレーム (stable)
        3. 6種類のカット画像を物体認識、ラベル付け
        4. カット画像とラベル結果を目視確認
        5. 統計的に最適なカット画像の選定方法を決める

        → 結果 lastをカット画像として採用
        　           | first | quarter | center | three_quarters | last | stable
        　ブレの回数 | 　12  |	  8	   |   14   |       7	     |   2	|    6

    Parameters
    ----------
    video_id : str
        動画ID 

    cut_point : list
        カット検出点（フレーム番号）のリスト
    
    frames : numpy.ndarray
        フレームデータ（動画の全画像データ） 
    
    dest_path : str
        保存先フォルダのパス
    """
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]   # RGBからBGRに戻す（戻さないと色が反転したまま保存される）

    cut_count = len(cut_point) # カット数
    for i in range(cut_count):
        last = cut_point[i] # カット画像

        # カット画像の保存
        cv2.imwrite(dest_path + '\\cut_img' + str(i+1) + '.jpg', frames[last])

    logger.debug(video_id + '_cut_img1 ～ ' + str(cut_count) + 'を保存しました')
    logger.debug('保存先 : ' + dest_path)
    logger.debug('--------------------------------------------------')

def cut_img_generate(video_path, result_cut_img_path):
    """
    各カットからカット画像を作成する関数
    カット画像 ・・・ 物体認識に使用する画像
    
    以下の流れで行う
    
    動画IDリストの作成
    カット画像の作成
        保存先フォルダの作成 
        動画の読み込み、フレームデータ，動画情報の抽出
        カット点の検出
        カット画像の保存

    Parameters
    ----------
    video_path : str
        入力する動画データのフォルダパス

    result_path
        カット分割結果を保存するフォルダパス
    """
    # --------------------------------------------------
    # 動画IDリストの作成
    # --------------------------------------------------
    files = os.listdir(video_path)  # 動画ファイル名（動画ID）一覧を取得
    video_id_list = [f.replace('.mp4', '') for f in files]  # 一覧から拡張子を削除

    # --------------------------------------------------
    # カット画像の作成
    # --------------------------------------------------
    for video_id in video_id_list:
        input_video_path = video_path + '\\' + video_id + '.mp4' # 動画ファイルの入力パス 
        
        # --------------------------------------------------
        # 保存先フォルダの作成
        # --------------------------------------------------
        dest_path = os.path.join(result_cut_img_path, video_id) # 各動画のカット画像作成結果の保存先
        dest_folder_create(dest_path)   # フォルダ作成 
        
        # --------------------------------------------------
        # 動画の読み込み、フレームデータと動画情報を抽出
        # --------------------------------------------------
        frames, video_info = read_video_data(input_video_path)
        
        # --------------------------------------------------
        # カット点の検出
        # --------------------------------------------------
        cut_point = cut_point_detect(frames)

        # --------------------------------------------------
        # カット画像の保存
        # --------------------------------------------------
        cut_img_save(video_id, cut_point, frames, dest_path)