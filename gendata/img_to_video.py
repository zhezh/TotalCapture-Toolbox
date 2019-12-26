import os
import subprocess


if __name__ == '__main__':
    base_path = '/data/extra/zhe/data/mvtc_fullgen_0124/marked/'
    dirs = os.listdir(base_path)
    for d in dirs:
        print(d)
        dir_path = os.path.join(base_path, d)
        cmd_cd = 'cd {}'.format(dir_path)
        cmd_convert = 'ffmpeg -r 60 -f image2 -i %06d.jpg -vcodec hevc_nvenc  -pix_fmt yuv420p ../{}.mp4'.format(d)
        cmd = cmd_cd + ' && ' + cmd_convert
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        break
