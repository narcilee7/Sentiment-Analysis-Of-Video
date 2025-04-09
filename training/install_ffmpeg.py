import subprocess
import sys


def install_ffmpeg():
    """安装和配置FFmpeg
    
    尝试通过多种方式安装FFmpeg：
    1. 通过pip安装ffmpeg-python包
    2. 下载并安装静态FFmpeg二进制文件
    
    Returns:
        bool: 安装是否成功
    """
    print("Starting FFmpeg installation...")

    # 更新pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # 更新setuptools
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])

    # 尝试通过pip安装ffmpeg-python
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("Installed ffmpeg-python successfully")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")
    
    # 尝试下载并安装静态FFmpeg二进制文件
    try:
        # 下载FFmpeg压缩包
        subprocess.check_call(['wget',
                            "https//johnansicke.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
                            "-O",
                            "/tmp/ffmpeg.tar.xz"
        ])

        # 解压FFmpeg
        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/",
        ])

        # 查找FFmpeg可执行文件
        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        )

        ffmpeg_path = result.stdout.strip()

        # 将FFmpeg复制到系统目录
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        # 设置执行权限
        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed static FFmpeg binary successfully")
    
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")

    # 验证安装
    try:
        # 检查FFmpeg版本
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("FFmpeg version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Failed installation verification:", e)
        return False