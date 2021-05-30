import os
import shutil
from natsort import natsorted
from file_io import dest_folder_create

# Cut_List, cut_videosを作成する
# Cut_List : カット（動画）を1つにまとめたフォルダ
# cut_videos : Cut_Listの動画名をまとめたテキストデータ
base = os.path.dirname(os.path.abspath(__file__))   # スクリプト実行ディレクトリ
cut_path = os.path.normpath(os.path.join(base, r'Result\Cut'))  # カット動画のパス
cut_list_path = os.path.normpath(os.path.join(base, r'Result\Cut_List'))  # カット動画のパス
cut_videos_path = os.path.normpath(os.path.join(base, r'Result\cut_videos'))  # カット動画のパス
video_id_list = os.listdir(cut_path)   # 動画IDリスト

# [Cut_List]の作成
dest_folder_create(cut_list_path)

# [cut_videos]の作成
f = open(cut_videos_path, 'w')

for video_id in video_id_list:
    videos_path = os.path.normpath(os.path.join(cut_path, video_id)) # カット動画フォルダのパス
    videos = natsorted(os.listdir(videos_path)) # カット動画リスト（自然順）
    for video in videos:

        new_file_name = video_id + '_' + video  # 新規ファイル名

        video_path = os.path.normpath(os.path.join(videos_path, video)) # カット動画のパス
        new_video_path = os.path.normpath(os.path.join(cut_list_path, new_file_name))   # 新規の保存先（Cut_List）

        # [cut_videos]に書き込み
        f.write(new_file_name + '\n')
        
        # [Cut_List]にコピー
        shutil.copyfile(video_path, new_video_path)
        
f.close()