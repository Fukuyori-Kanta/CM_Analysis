import os
import os.path
import configparser
import errno

def set_path(ini_file):
    """
    設定ファイルからパスを取得して設定する関数
    各パスをリストにまとめて返す

    Parameters
    ----------
    ini_file : str
        設定ファイルのパス

    Returns
    -------
    path : list
        設定したパスのリスト[root_path, video_path, cmData_top_path, cmData_btm_path, cut_path, cut_img_path]
    """
    # --------------------------------------------------
    # 設定ファイルの読み込み
    # --------------------------------------------------
    config = configparser.ConfigParser()

    # ファイル存在チェック
    if not os.path.exists(ini_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ini_file)
    
    config.read(ini_file, encoding='utf-8')

    # --------------------------------------------------
    # 設定ファイルから値を設定
    # --------------------------------------------------
    # パスの設定
    root_path = config['PATH']['root_path']                                         # ルートパス
    video_path = os.path.join(root_path, config['PATH']['video_path'])              # 動画データのパス 
    cmData_top_path = os.path.join(root_path, config['PATH']['cmData_top_path'])    # CMデータ(上位1000)のファイルパス
    cmData_btm_path = os.path.join(root_path, config['PATH']['cmData_btm_path'])    # CMデータ(下位1000)のファイルパス
    cut_path = os.path.join(root_path, config['PATH']['cut_path'])                  # カット分割結果の保存フォルダパス
    cut_img_path = os.path.join(root_path, config['PATH']['cut_img_path'])          # カット画像作成結果の保存フォルダパス
    cut_point_path = os.path.join(root_path, config['PATH']['cut_point_path'])      # カット点データ（.csv）の保存ファイルパス

    return [root_path, video_path, cmData_top_path, cmData_btm_path, cut_path, cut_img_path, cut_point_path]