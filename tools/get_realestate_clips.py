import argparse
import json
import os
import os.path as osp
from tqdm import tqdm
from moviepy import VideoFileClip
import imageio
from decord import VideoReader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blob_path', required=True)
    parser.add_argument('--video_root', default="data/RealEstate10K_720p/__cache__/")
    parser.add_argument('--save_path', default="data/RealEstate10K_CamCtrl/train/")
    parser.add_argument('--video2clip_json', default="data/RealEstate10K_raw/train_video2clip.json")
    parser.add_argument('--clip_txt_path', default="data/RealEstate10K_raw/train/")
    parser.add_argument('--rank', type=int, default=0, help='used for parallel processing')
    parser.add_argument('--world_size', type=int, default=1, help='used for parallel processing')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.save_path = osp.join(args.blob_path, args.save_path)
    args.video_root = osp.join(args.blob_path, args.video_root)
    args.clip_txt_path = osp.join(args.blob_path, args.clip_txt_path)
    args.video2clip_json = osp.join(args.blob_path, args.video2clip_json)
    
    
    os.makedirs(args.save_path, exist_ok=True)
    video2clips = json.load(open(args.video2clip_json, 'r'))
    
    len_videos = len(video2clips)
    low_idx = len_videos * args.rank // args.world_size
    high_idx = len_videos * (args.rank + 1) // args.world_size
    video_names = list(video2clips.keys())[low_idx: high_idx] if high_idx != -1 else list(video2clips.keys())
    video2clips = {k: v for k, v in video2clips.items() if k in video_names}
    support_formats = ["mp4", "mkv", "webm", "avi", "mov"]
    for video_name, clip_list in tqdm(video2clips.items()):
        # TRY TO FIND THE VIDEO FILE
        video_path = ''
        for fmt in support_formats:
            video_path = osp.join(args.video_root, video_name + '.' + fmt)
            if osp.exists(video_path):
                break
        if not osp.exists(video_path):
            continue
        video = VideoFileClip(video_path)
        clip_save_path = osp.join(args.save_path, video_name)
        os.makedirs(clip_save_path, exist_ok=True)
        for clip in tqdm(clip_list):
            clip_save_name = clip + '.mp4'
            if osp.exists(osp.join(clip_save_path, clip_save_name)):
                continue
            with open(osp.join(args.clip_txt_path, clip + '.txt'), 'r') as f:
                lines = f.readlines()
            frames = [x for x in lines[1: ]]
            timesteps = [int(x.split(' ')[0]) for x in frames]
            if timesteps[-1] <= timesteps[0]:
                continue
            timestamps_seconds = [x / 1000000.0 for x in timesteps]
            frames = [video.get_frame(t) for t in timestamps_seconds]
            imageio.mimsave(osp.join(clip_save_path, clip_save_name), frames, fps=video.fps)
            video_reader = VideoReader(osp.join(clip_save_path, clip_save_name))
            assert len(video_reader) == len(timesteps)
