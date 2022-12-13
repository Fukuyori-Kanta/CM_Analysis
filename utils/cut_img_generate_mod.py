import cv2
import os
import ast
from utils.init_setting import setup_logger
from utils.file_io import create_dest_folder, read_csv
from utils.cut_segmentation_mod import read_video_data

CUT_EXTENSION = '.mp4'          # 保存するカットの拡張子（MP4）
CUT_IMG_EXTENSION = '.jpg'      # 保存するカット画像の拡張子（JPEG）

# ログ設定
logger = setup_logger(__name__)

def save_cut_img(video_id, cut_point, frames, dest_path):
    """カット範囲から一番最後のフレームをカット画像として保存する
    
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
         ブレの回数 |   12  |	  8	  |   14   |       7	    |   2  |    6

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
        save_cut_img_path = os.path.normpath(os.path.join(dest_path, 'cut_img' + str(i+1) + CUT_IMG_EXTENSION))    # 保存先
        cv2.imwrite(save_cut_img_path, frames[last])

    logger.debug(video_id + '_cut_img1 ～ ' + str(cut_count) + 'を保存しました')
    logger.debug('保存先 : ' + dest_path)
    logger.debug('-' * 90)

def cut_img_generate(video_path, cut_img_path, cut_point_path):
    """各カットからカット画像を作成する関数

    カット画像 ・・・ 物体認識に使用する画像
    
    [手順]
        1. 保存先フォルダの作成
        2. カット点データの読み込み 
        3. 動画の読み込み、フレームデータ，動画情報の抽出
        4. カット画像の保存

    Parameters
    ----------
    video_path : str
        入力する動画データのフォルダパス

    cut_img_path : str
        分割したカット画像を保存するフォルダパス
    
    cut_point_path : str
        分割したカット点を保存しているファイルパス（.csv）   
        [cut_segmentation_mod.py] で行った結果    
    """
    # --------------------------------------------------
    # カット画像の保存先フォルダの作成
    # --------------------------------------------------
    create_dest_folder(cut_img_path)    # フォルダ作成
    
    # --------------------------------------------------
    # 動画ID一覧とカット点の読み込み
    # --------------------------------------------------
    video_id_list, cut_point_str = read_csv(cut_point_path)
    cut_point_list = [ast.literal_eval(str) for str in cut_point_str]   # 文字列になっているのでリストに直す
    
    # --------------------------------------------------
    # カット画像の作成
    # --------------------------------------------------
    for video_id, cut_point in zip(video_id_list, cut_point_list):
        file_name = video_id + CUT_EXTENSION    # ファイル名.拡張子
        input_video_path = os.path.normpath(os.path.join(video_path, file_name)) # 動画ファイルの入力パス 

        # --------------------------------------------------
        # 保存先フォルダの作成
        # --------------------------------------------------
        dest_path = os.path.join(cut_img_path, video_id) # 各動画のカット画像作成結果の保存先
        create_dest_folder(dest_path)   # フォルダ作成 
        
        # --------------------------------------------------
        # 動画の読み込み、フレームデータを抽出
        # --------------------------------------------------
        frames, _ = read_video_data(input_video_path)
    
        # --------------------------------------------------
        # カット画像の保存
        # --------------------------------------------------
        save_cut_img(video_id, cut_point, frames, dest_path)