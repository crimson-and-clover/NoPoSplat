import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm

from multiprocessing import Pool

TARGET_BYTES_PER_CHUNK = int(1e8)


def process(data, seq_id, videoname, output_root):
    seqname = data["seq_names"][seq_id]
    save_path = os.path.join(output_root, seqname, "data.npz")

    os.makedirs(os.path.join(output_root, seqname), exist_ok=True)

    if os.path.exists(save_path):
        return

    for timestamp in data["list_timestamps"][seq_id]:
        output_image = os.path.join(output_root, seqname, f'{timestamp}.jpg')
        if os.path.exists(output_image):
            continue
        command = f'ffmpeg -hide_banner -loglevel error -y -ss {timestamp / 1000000.0:.3f} -i {videoname} -vframes 1 -q:v 1 -f image2 {output_image}'
        os.system(command)

    png_list = glob.glob(os.path.join(output_root, seqname, "*.jpg"))
    images = {}

    for pngname in png_list:
        image = None
        with open(pngname, mode="rb") as input:
            image = np.frombuffer(input.read(), dtype=np.uint8)
        basefile = os.path.basename(pngname)
        images[basefile] = image
        os.remove(pngname)

    np.savez(save_path, **images)


def process_wrapper(args):
    return process(*args)


class DataExtractor:
    def __init__(self, dataroot, output_root, mode="test"):
        self.dataroot = dataroot
        self.output_root = os.path.join(output_root, mode)
        self.mode = mode
        self.list_seqnames = sorted(glob.glob(os.path.join(dataroot, "*.txt")))

        os.makedirs(self.output_root, exist_ok=True)

        self.list_data = []
        for txt_file in tqdm(self.list_seqnames, desc="Loading txt Files", unit="file"):
            basename = os.path.basename(txt_file)
            seq_name = os.path.splitext(basename)[0]
            lines = []
            with open(txt_file, "r") as reader:
                lines = reader.readlines()
            youtube_url = lines[0].strip()
            list_timestamps = list(map(
                lambda line: int(line.split(' ')[0].strip()), lines[1:]
            ))
            register = list(filter(
                lambda x: x["youtube_url"] == youtube_url, self.list_data
            ))
            if len(register) == 0:
                self.list_data.append({
                    "youtube_url": youtube_url,
                    "seq_names": [seq_name],
                    "list_timestamps": [list_timestamps],
                })
            else:
                register[0]["seq_names"].append(seq_name)
                register[0]["list_timestamps"].append(list_timestamps)

    def Run(self):
        with Pool(processes=4) as pool:
            params = []
            for data in tqdm(self.list_data, desc="Loading Parameters"):
                hash_code = data["youtube_url"].split("?v=")[1]
                videoname = os.path.join(
                    self.dataroot, "videos", f"{hash_code}.mp4"
                )
                if not os.path.exists(videoname):
                    print(f"video {videoname} not found")
                    exit(1)
                params.extend(map(lambda i: (data, i, videoname, self.output_root),
                                  range(len(data["seq_names"]))))
            results = []
            for result in tqdm(
                pool.imap_unordered(process_wrapper, params),
                total=len(params),
                desc="Extract Key Frame"
            ):
                results.append(result)

    def Show(self):
        print("########################################")
        total_sum = sum(
            map(lambda x: len(x),
                map(lambda d: d["list_timestamps"], self.list_data)
                ))
        for data in self.list_data:
            timestamps_sum = sum(
                map(lambda x: len(x), data["list_timestamps"]))
            print(f' URL : {data["youtube_url"]}')
            print(f' SEQ : {len(data["seq_names"])}')
            print(f' LEN : {timestamps_sum}')
            print("----------------------------------------")
        print(f"TOTAL : {total_sum} sequnces")


def load_data_to_tensor(filename):
    data = np.load(filename)
    tensors = {
        int(os.path.splitext(key)[0]): torch.tensor(data[key]) for key in data
    }
    return tensors


def load_meta_data(filename):
    lines = []
    with open(filename, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
    url = lines[0].strip()

    timestamps = []
    cameras = []

    for line in lines[1:]:
        timestamp, *camera = line.strip().split(" ")
        timestamps.append(int(timestamp))
        cameras.append(np.fromstring(",".join(camera), sep=","))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


class DataConverter:
    def __init__(self, dataroot, output_root, mode="test"):
        self.dataroot = dataroot
        self.output_root = os.path.join(output_root, mode)
        self.mode = mode
        os.makedirs(self.output_root, exist_ok=True)

    def convert(self):
        metadata_files = sorted(
            glob.glob(os.path.join(self.dataroot, "*.txt")))
        chunk = []
        chunk_size = 0
        chunk_index = 0
        for txt_file in tqdm(metadata_files, desc="Converting Files", unit="file"):
            basename = os.path.basename(txt_file)
            seq_name = os.path.splitext(basename)[0]
            images = load_data_to_tensor(os.path.join(
                self.output_root, seq_name, "data.npz"))
            metadata = load_meta_data(txt_file)
            assert len(images) == len(metadata["timestamps"])
            metadata["images"] = [
                images[timestamp.item()] for timestamp in metadata["timestamps"]
            ]
            metadata["key"] = seq_name
            image_size = sum(map(lambda x: x.shape[0], metadata["images"]))
            chunk.append(metadata)
            chunk_size += image_size

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                torch.save(chunk, os.path.join(
                    self.output_root, f"{chunk_index:0>6}.torch"))
                chunk_size = 0
                chunk_index += 1
                chunk = []

        if chunk_size > 0:
            torch.save(chunk, os.path.join(
                self.output_root, f"{chunk_index:0>6}.torch"))
            chunk_size = 0
            chunk_index += 1
            chunk = []


def main():
    if len(sys.argv) != 4:
        print("usage: this.py [test or train] [dataroot] [output_root]")
        exit(1)
    if sys.argv[1] == "test":
        mode = "test"
    elif sys.argv[1] == "train":
        mode = "train"
    else:
        print("invalid mode")
        exit(1)

    dataroot = os.path.abspath(os.path.join(sys.argv[2], mode))
    output_root = os.path.abspath(sys.argv[3])

    # video to data.npz
    extractor = DataExtractor(dataroot, output_root, mode)
    extractor.Show()
    extractor.Run()

    # data.npz and meta to %06d.torch
    converter = DataConverter(dataroot, output_root, mode)
    converter.convert()


if __name__ == "__main__":
    main()
