import cv2
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_videos_to_hdf5(video_dir: str, output_file: str, img_size: int = 224):
    video_dir = Path(video_dir)
    output_file = Path(output_file)
    
    # 如果输出文件夹 datasets/ 不存在，自动创建
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 mp4 视频（如果有其他格式，可以加进来）
    video_paths = list(video_dir.glob("*.mp4"))
    if not video_paths:
        print(f"❌ 在 {video_dir} 文件夹下没有找到 mp4 视频！")
        return
        
    print(f"找到 {len(video_paths)} 个视频，准备开始转换...")
    
    # 打开/创建 HDF5 文件
    with h5py.File(output_file, 'w') as f:
        # 创建可动态扩展的数据集（maxshape设为None，代表行数可以无限增加）
        pixel_ds = f.create_dataset('pixels', shape=(0, 3, img_size, img_size), 
                                    maxshape=(None, 3, img_size, img_size), 
                                    dtype='uint8', chunks=True)
        ep_idx_ds = f.create_dataset('episode_idx', shape=(0,), maxshape=(None,), dtype='int32')
        step_idx_ds = f.create_dataset('step_idx', shape=(0,), maxshape=(None,), dtype='int32')
        
        total_frames_written = 0
        
        # 遍历每个短视频
        for ep_id, v_path in enumerate(tqdm(video_paths, desc="处理视频中")):
            cap = cv2.VideoCapture(str(v_path))
            
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 1. 改变图像大小
                frame = cv2.resize(frame, (img_size, img_size))
                # 2. OpenCV 默认是 BGR，深度学习通常用 RGB，进行转换
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 3. 把 [高度, 宽度, 通道数] 变成 PyTorch 喜欢的 [通道数, 高度, 宽度]
                frame = np.transpose(frame, (2, 0, 1))
                
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                continue
                
            # 转换为 numpy 数组
            frames = np.array(frames, dtype=np.uint8)
            num_frames = len(frames)
            
            # 扩展 HDF5 里的数据集容量，装下新读入的这个视频
            pixel_ds.resize(total_frames_written + num_frames, axis=0)
            ep_idx_ds.resize(total_frames_written + num_frames, axis=0)
            step_idx_ds.resize(total_frames_written + num_frames, axis=0)
            
            # 把数据写入 HDF5
            pixel_ds[total_frames_written:] = frames
            ep_idx_ds[total_frames_written:] = np.full((num_frames,), ep_id, dtype=np.int32)
            step_idx_ds[total_frames_written:] = np.arange(num_frames, dtype=np.int32)  # 0, 1, 2, 3...
            
            total_frames_written += num_frames
            
    print(f"\n✅ 转换圆满成功！")
    print(f"📁 文件已保存至: {output_file}")
    print(f"🎬 共处理视频: {len(video_paths)} 个")
    print(f"🖼️ 共存储帧数: {total_frames_written} 帧")

if __name__ == "__main__":
    # 使用方法：
    # 1. 在项目目录下建一个 raw_videos 文件夹，把短视频集锦放进去
    # 2. 运行此脚本，它会自动在 datasets 文件夹下生成 hdf5
    convert_videos_to_hdf5("./raw_videos", "./datasets/wzry_unsupervised.hdf5")