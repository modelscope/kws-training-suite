import argparse
import os
import wave
from concurrent import futures

from tqdm import tqdm

NUM_THS = 1


def list_files(path, ext, abs_path=True):
    path_length = len(path)
    if not path.endswith('/'):
        path_length += 1
    for root, dirs, files in os.walk(path, followlinks=True):
        for file in files:
            if file.endswith(ext):
                if abs_path:
                    yield os.path.join(root, file)
                else:
                    yield os.path.join(root, file)[path_length:]


def split(raw_wav, base_in, base_out, duration):
    rel_path = os.path.relpath(raw_wav, base_in)
    rel_dirs = os.path.dirname(rel_path)
    output_dir = os.path.join(base_out, rel_dirs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    raw_base, ext = os.path.splitext(os.path.basename(raw_wav))
    # file to extract the snippet from
    with wave.open(raw_wav, "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        frame_count = int(framerate * duration)
        expected_length = frame_count * sampwidth
        index = 0
        while True:
            # extract data
            data = infile.readframes(frame_count)
            if len(data) == 0:
                break
            index += 1
            out_path = os.path.join(output_dir,
                                    f'{raw_base}_seg{index:04d}{ext}')
            # write the extracted data to a new file
            with wave.open(out_path, 'wb') as outfile:
                outfile.setnchannels(nchannels)
                outfile.setsampwidth(sampwidth)
                outfile.setframerate(framerate)
                outfile.setnframes(frame_count)
                outfile.writeframes(data)
            if len(data) != expected_length:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to split *.wav file')
    parser.add_argument('input', help='path of the audio list file or the directory storing audio files')
    parser.add_argument('-d', '--duration', type=int, default=60, help='the duration of new wav file in second')
    parser.add_argument('-o', '--out_dir', help='output directory, default: [input]_seg[duration]')
    parser.add_argument('-t', '--threads', type=int, help='parallel thread number, default: 1')
    args = parser.parse_args()

    threads = args.threads if args.threads else NUM_THS
    duration = args.duration
    basein = os.path.abspath(args.input)
    if args.out_dir:
        baseout = args.out_dir
    else:
        baseout = f'{basein}_seg{duration}'
    os.makedirs(baseout)

    if os.path.isdir(basein):
        flist = list_files(basein, '.wav')
    else:
        with open(basein, 'r', encoding='UTF-8') as fd:
            flist = fd.readlines()

    tasks = []
    with futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for f in flist:
            tasks.append(executor.submit(split, f, basein, baseout, duration))
        for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
            pass
